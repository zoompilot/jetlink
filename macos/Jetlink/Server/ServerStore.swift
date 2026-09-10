import Foundation
import Observation
import os

enum ServerRunState: Equatable, Sendable {
  case stopped
  case starting
  case serving
  case stopping
  case failed(String)
}

struct ServerInfo: Equatable, Sendable {
  let pid: Int32
  let version: String
  let python: String
  let backend: String
  let runtimeVersion: String
  let device: String
  let cache: String
  let transport: String
  let port: Int?

  init(pid: Int32, version: String, python: String, backend: String, runtimeVersion: String, device: String, cache: String, transport: String, port: Int?) {
    self.pid = pid
    self.version = version
    self.python = python
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
    self.cache = cache
    self.transport = transport
    self.port = port
  }
}

enum ServerStoreError: Error, LocalizedError, Equatable {
  case notRunning
  case failed(String)

  var errorDescription: String? {
    switch self {
    case .notRunning: return "The server is not running."
    case .failed(let detail): return detail
    }
  }
}

/// Owns the server process, the control connection, and everything the Status
/// view shows.
@MainActor
@Observable
final class ServerStore {
  private(set) var runState: ServerRunState = .stopped
  private(set) var info: ServerInfo?
  private(set) var link: LinkEvent = .waiting
  private(set) var engine: EngineEvent = .none
  private(set) var stats: StatsEvent?
  private(set) var startedAt: Date?
  var lastFailure: String?

  let settings: AppSettings
  let logs: LogBuffer
  /// Every event a `ModelStore` cares about: inventory, catalog, download, import.
  let modelEvents: AsyncStream<ControlEvent>

  @ObservationIgnored private let modelEventsContinuation: AsyncStream<ControlEvent>.Continuation
  @ObservationIgnored private let logFile: LogFileWriter?
  @ObservationIgnored private let sleepAssertion: SleepAssertion
  @ObservationIgnored private let process = ServerProcess()
  @ObservationIgnored private let isLive: Bool
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "server")
  @ObservationIgnored private var client: ControlClient?
  @ObservationIgnored private var connectTask: Task<Void, Never>?
  @ObservationIgnored private var consumeTask: Task<Void, Never>?
  @ObservationIgnored private var restartTask: Task<Void, Never>?
  @ObservationIgnored private var stopRequested = false
  /// Set while `handleConnectFailure` is taking the process down itself, so the
  /// exit it causes is not mistaken for a crash. Cleared by the next `start()`.
  @ObservationIgnored private var suppressExitHandling = false
  @ObservationIgnored private var restartTimes: [Date] = []

  private static let backoff: [Duration] = [.seconds(5), .seconds(10), .seconds(20), .seconds(40), .seconds(60)]
  private static let restartWindow: TimeInterval = 600
  private static let maxRestarts = 5

  init(settings: AppSettings, logs: LogBuffer, logFile: LogFileWriter? = LogFileWriter(), isLive: Bool = true) {
    self.settings = settings
    self.logs = logs
    self.logFile = logFile
    self.isLive = isLive
    self.sleepAssertion = SleepAssertion()
    let (stream, continuation) = AsyncStream<ControlEvent>.makeStream(bufferingPolicy: .unbounded)
    self.modelEvents = stream
    self.modelEventsContinuation = continuation
    if isLive {
      sleepAssertion.onPowerSourceChange = { [weak self] in self?.updateSleepAssertion() }
      sleepAssertion.startObservingPowerSource()
    }
  }

  // MARK: lifecycle

  func start() {
    guard isLive else { return }
    switch runState {
    case .stopped, .failed: break
    default: return
    }
    runState = .starting
    lastFailure = nil
    stopRequested = false
    suppressExitHandling = false

    let runtime: PythonRuntime
    switch PythonRuntime.locate(settings: settings) {
    case .success(let located):
      runtime = located
    case .failure(let error):
      failStartup(error.localizedDescription)
      return
    }

    let socket = AppSettings.controlSocketURL
    try? FileManager.default.removeItem(at: socket)
    let configuration = ServerConfiguration(
      python: runtime.executable,
      backend: settings.backend,
      transport: settings.transport,
      tcpPort: settings.tcpPort,
      cacheDirectory: settings.cacheDirectory,
      controlSocket: socket,
      logLevel: settings.logLevel,
      logFile: AppSettings.logFileURL)

    let buffer = logs
    let file = logFile
    process.onLine = { line in
      Task { @MainActor in buffer.append(line) }
      if let file { Task { await file.append(line) } }
    }
    process.onExit = { [weak self] status, reason in
      Task { @MainActor in self?.handleExit(status: status, reason: reason) }
    }

    do {
      try process.start(configuration)
    } catch {
      failStartup(error.localizedDescription)
      return
    }

    let client = ControlClient(socketPath: socket)
    self.client = client
    connectTask = Task { [weak self] in
      do {
        // A CoreML probe can hold the server for a while before it listens.
        try await client.connect(retryingFor: .seconds(120))
      } catch {
        await self?.handleConnectFailure(error)
        return
      }
      self?.startConsuming(client)
    }
    updateSleepAssertion()
  }

  func stop() {
    Task { await stopAndWait() }
  }

  /// The awaitable form of `stop()`, used when the app is quitting.
  func stopAndWait() async {
    guard isLive else { return }
    restartTask?.cancel()
    restartTask = nil
    if case .stopped = runState { return }
    runState = .stopping
    stopRequested = true
    if let client, client.isConnected {
      _ = try? await client.send(.shutdown, timeout: .seconds(3))
    }
    await process.stop()
    cancelTasks()
    client?.close()
    client = nil
    runState = .stopped
    resetLiveState()
    updateSleepAssertion()
  }

  func restart() {
    Task {
      await stopAndWait()
      start()
    }
  }

  func send(_ command: ControlCommand) async throws -> ReplyEvent {
    guard let client, client.isConnected else { throw ServerStoreError.notRunning }
    return try await client.send(command)
  }

  /// Starts the server if it is not serving, and waits until it is.
  func startIfNeeded() async throws {
    guard isLive else { throw ServerStoreError.notRunning }
    if case .serving = runState { return }
    switch runState {
    case .stopped, .failed: start()
    default: break
    }
    let deadline = Date().addingTimeInterval(180)
    while Date() < deadline {
      switch runState {
      case .serving: return
      case .failed(let detail): throw ServerStoreError.failed(detail)
      default: break
      }
      try? await Task.sleep(for: .milliseconds(100))
    }
    throw ServerStoreError.failed("The server did not start in time.")
  }

  // MARK: events

  private func startConsuming(_ client: ControlClient) {
    consumeTask?.cancel()
    consumeTask = Task { [weak self] in
      for await event in client.events {
        if Task.isCancelled { break }
        self?.apply(event)
      }
    }
  }

  func apply(_ event: ControlEvent) {
    switch event {
    case .hello(let hello):
      info = ServerInfo(
        pid: hello.pid,
        version: hello.version,
        python: hello.python,
        backend: info?.backend ?? "",
        runtimeVersion: info?.runtimeVersion ?? "",
        device: info?.device ?? "",
        cache: hello.cache,
        transport: hello.transport,
        port: hello.port)
      runState = .serving
      startedAt = Date()
      pruneRestartHistory()
      updateSleepAssertion()
    case .server(let server):
      if let current = info {
        info = ServerInfo(
          pid: current.pid,
          version: current.version,
          python: current.python,
          backend: server.backend ?? current.backend,
          runtimeVersion: server.runtimeVersion ?? current.runtimeVersion,
          device: server.device ?? current.device,
          cache: current.cache,
          transport: current.transport,
          port: current.port)
      }
      if server.state == "stopping" { stopRequested = true }
    case .link(let value):
      link = value
      if value.state != .connected { stats = nil }
    case .engine(let value):
      engine = value
    case .stats(let value):
      stats = value
    case .reply:
      break
    default:
      modelEventsContinuation.yield(event)
    }
  }

  private func handleConnectFailure(_ error: any Error) async {
    guard !stopRequested else { return }
    log.error("could not reach the control channel: \(error.localizedDescription, privacy: .public)")
    // A server that never listened is still holding the GPU and the USB device,
    // so take it down before showing the failure.
    suppressExitHandling = true
    await process.stop()
    cancelTasks()
    client?.close()
    client = nil
    let detail = "The server started but its control channel never answered."
    lastFailure = detail
    runState = .failed(detail)
    resetLiveState()
    updateSleepAssertion()
  }

  private func handleExit(status: Int32, reason: Process.TerminationReason) {
    guard !suppressExitHandling else { return }
    cancelTasks()
    client?.close()
    client = nil
    if stopRequested {
      runState = .stopped
      resetLiveState()
      updateSleepAssertion()
      return
    }
    let wasStarting: Bool
    if case .starting = runState { wasStarting = true } else { wasStarting = false }
    let message = ServerStore.exitMessage(status: status, reason: reason)
    lastFailure = message
    runState = .failed(message)
    resetLiveState()
    updateSleepAssertion()
    // A failure during startup is almost always configuration, so it is not retried.
    guard !wasStarting else { return }
    scheduleRestart()
  }

  /// One sentence on how the server went, with the signal by name. The last
  /// log lines are not in it: the Status view shows them underneath.
  nonisolated static func exitMessage(status: Int32, reason: Process.TerminationReason) -> String {
    guard reason == .uncaughtSignal else {
      return "The server exited with status \(status)."
    }
    let names: [Int32: String] = [
      SIGKILL: "SIGKILL", SIGSEGV: "SIGSEGV", SIGABRT: "SIGABRT", SIGBUS: "SIGBUS",
      SIGILL: "SIGILL", SIGTRAP: "SIGTRAP", SIGTERM: "SIGTERM", SIGINT: "SIGINT",
    ]
    let name = names[status].map { " (\($0))" } ?? ""
    var text = "The server was killed by signal \(status)\(name)."
    if [SIGKILL, SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGTRAP].contains(status) {
      text += " Console may have a crash report for python3.14 under Crash Reports."
    }
    return text
  }

  private func failStartup(_ detail: String) {
    lastFailure = detail
    runState = .failed(detail)
    updateSleepAssertion()
  }

  private func scheduleRestart() {
    pruneRestartHistory()
    guard restartTimes.count < ServerStore.maxRestarts else {
      let detail = "The server failed five times in ten minutes. Jetlink stopped restarting it."
      lastFailure = detail
      runState = .failed(detail)
      return
    }
    let delay = ServerStore.backoff[min(restartTimes.count, ServerStore.backoff.count - 1)]
    restartTimes.append(Date())
    restartTask?.cancel()
    restartTask = Task { [weak self] in
      try? await Task.sleep(for: delay)
      guard !Task.isCancelled else { return }
      self?.start()
    }
  }

  private func pruneRestartHistory() {
    let now = Date()
    restartTimes = restartTimes.filter { now.timeIntervalSince($0) < ServerStore.restartWindow }
  }

  private func cancelTasks() {
    connectTask?.cancel()
    connectTask = nil
    consumeTask?.cancel()
    consumeTask = nil
  }

  private func resetLiveState() {
    link = .waiting
    engine = .none
    stats = nil
    startedAt = nil
  }

  private func updateSleepAssertion() {
    guard isLive else { return }
    let wanted: Bool
    if case .serving = runState {
      wanted = settings.keepAwakeWhileServing && sleepAssertion.isOnACPower
    } else {
      wanted = false
    }
    sleepAssertion.setActive(wanted)
  }

  /// Called by the settings view when keepAwakeWhileServing changes.
  func keepAwakeSettingChanged() {
    updateSleepAssertion()
  }
}

extension ServerStore {
  /// A store with fixed state and nothing behind it. Actions are no-ops.
  static func preview(
    runState: ServerRunState = .serving,
    info: ServerInfo? = nil,
    link: LinkEvent,
    engine: EngineEvent,
    stats: StatsEvent? = nil
  ) -> ServerStore {
    let store = ServerStore(settings: AppSettings.preview(), logs: LogBuffer(), logFile: nil, isLive: false)
    store.runState = runState
    store.info = info
    store.link = link
    store.engine = engine
    store.stats = stats
    store.startedAt = Date(timeIntervalSinceNow: -3600)
    return store
  }
}
