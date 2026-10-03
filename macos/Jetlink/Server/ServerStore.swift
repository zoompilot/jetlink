import Foundation
import JetlinkKit
import JetlinkORT
import JetlinkServer
import Observation
import os

/// What the running server is, for Status and the menu bar: the choice it
/// was started with, and what its backend reports once it is up.
struct ServerInfo: Equatable, Sendable {
  let version: String
  let choice: BackendChoice
  var runtimeVersion: String
  var device: String
  let cache: String
  let transport: String
  let port: Int?
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

/// Owns the server, and everything the Status view shows. The server is the
/// Swift one the iPhone app runs, in this process through `EmbeddedServer`:
/// nothing to find or start, and the control channel's events without a
/// socket. It cannot crash apart from the app, so there is nothing to restart
/// behind the user's back.
@MainActor
@Observable
final class ServerStore: ServerControlling {
  private(set) var runState: ServerRunState = .stopped
  private(set) var info: ServerInfo?
  /// What the screens show of the server, kept from its events.
  let state = ServerViewState()
  var link: LinkEvent { state.link }
  var engine: EngineEvent { state.engine }
  private(set) var startedAt: Date?
  var lastFailure: String?

  let settings: AppSettings
  let logs: LogBuffer
  /// Every event a `ModelStore` cares about: inventory, catalog, download, import.
  let modelEvents: AsyncStream<ControlEvent>

  @ObservationIgnored private let modelEventsContinuation: AsyncStream<ControlEvent>.Continuation
  @ObservationIgnored private let logFile: LogFileWriter?
  @ObservationIgnored private let sleepAssertion: SleepAssertion
  @ObservationIgnored private let isLive: Bool
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "server")
  @ObservationIgnored private var embedded: EmbeddedServer?
  /// Building and starting the server, off the main thread.
  @ObservationIgnored private var startTask: Task<Void, Never>?
  @ObservationIgnored private var consumeTask: Task<Void, Never>?
  @ObservationIgnored private var logStream: LogStream?
  @ObservationIgnored private var logTask: Task<Void, Never>?

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
    // One ordered stream of log lines, drained by one task into the Logs
    // view and server.log; a task per line could reorder them.
    let stream = LogStream()
    logStream = stream
    let buffer = logs
    let file = logFile
    logTask = Task {
      for await line in stream.lines {
        buffer.append(line)
        await file?.append(line)
      }
    }
    let configuration = ServerStore.configuration(transport: settings.transport, tcpPort: settings.tcpPort, cacheDirectory: settings.cacheDirectory)
    let choice = settings.backend
    // The controller's first .server event fills in the backend fields.
    let seed = ServerInfo(
      version: ServerStore.appVersion, choice: settings.backend, runtimeVersion: "", device: "", cache: settings.cacheDirectory.path(percentEncoded: false),
      transport: settings.transport.rawValue, port: configuration.listen ? Int(configuration.port) : nil)
    // Creating the cache, the engine cache's first look at the disk and the
    // first inventory are file work the main thread should not wait on.
    startTask = Task { [weak self] in
      let started = await Task.detached(priority: .userInitiated) {
        Result {
          let embedded = try EmbeddedServer(configuration: configuration, backend: ServerStore.backend(for: choice), gadget: USBGadget())
          try embedded.start()
          return embedded
        }
      }.value
      guard let self else { return }
      self.startTask = nil
      switch started {
      case .success(let embedded):
        self.embedded = embedded
        self.info = seed
        self.consumeTask = Task { [weak self] in
          for await event in embedded.events {
            self?.apply(event)
          }
        }
        // A stop that came while starting is waiting on this task, and
        // stops the server itself.
        if self.runState == .starting {
          self.runState = .serving
          self.startedAt = Date()
        }
      case .failure(let error):
        self.tearDown()
        let detail = "The server could not start: \(error)"
        self.log.error("\(detail, privacy: .public)")
        self.lastFailure = detail
        self.runState = .failed(detail)
      }
      self.updateSleepAssertion()
    }
  }

  func stop() {
    Task { await stopAndWait() }
  }

  /// Stops the server and lets the engine go, off the main thread: releasing
  /// a CoreML model can take a moment. The app awaits this when it quits.
  func stopAndWait() async {
    guard isLive else { return }
    switch runState {
    case .stopped, .stopping: return
    default: break
    }
    runState = .stopping
    await startTask?.value
    if let embedded {
      await Task.detached { embedded.stop(releasingEngine: true) }.value
    }
    tearDown()
    runState = .stopped
    updateSleepAssertion()
  }

  private func tearDown() {
    embedded = nil
    consumeTask?.cancel()
    consumeTask = nil
    logStream?.finish()
    logStream = nil
    logTask = nil
    resetLiveState()
  }

  func restart() {
    Task {
      await stopAndWait()
      start()
    }
  }

  func send(_ command: ControlCommand) async throws -> ReplyEvent {
    guard let embedded else { throw ServerStoreError.notRunning }
    return await embedded.handle(command)
  }

  /// Starts the server if it is not serving, and waits until it is.
  func startIfNeeded() async throws {
    guard isLive else { throw ServerStoreError.notRunning }
    if case .serving = runState { return }
    switch runState {
    case .stopped, .failed: start()
    default: break
    }
    let deadline = Date().addingTimeInterval(30)
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

  func apply(_ event: ControlEvent) {
    switch event {
    case .server(let server):
      if let version = server.runtimeVersion { info?.runtimeVersion = version }
      if let device = server.device { info?.device = device }
    case .shutdownRequest(let value):
      log.warning("the comma asked this Mac to power off (\(value.reason, privacy: .public)); a Mac does not")
    default:
      break
    }
    if !state.apply(event) {
      modelEventsContinuation.yield(event)
    }
  }

  // MARK: configuration

  /// What the server is asked to be, from the settings.
  nonisolated static func configuration(transport: TransportChoice, tcpPort: Int, cacheDirectory: URL) -> Server.Configuration {
    let port = UInt16(clamping: tcpPort > 0 ? tcpPort : AppSettings.defaultTCPPort)
    return Server.Configuration(port: port, cacheRoot: cacheDirectory, preload: true, listen: transport == .tcp, usb: transport == .usb)
  }

  /// What runs the model: CoreML on the device the setting names, with the
  /// GPU and a CPU core kept up between frames.
  nonisolated static func backend(for choice: BackendChoice) -> OrtBackend {
    OrtBackend(profile: choice.profile, preparer: ONNXPreparer(), keepAlive: true, keepCPUWarm: true)
  }

  nonisolated static var appVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
  }

  private func resetLiveState() {
    state.serverStopped()
    startedAt = nil
  }

  private func updateSleepAssertion() {
    guard isLive else { return }
    let wanted: Bool
    if case .serving = runState {
      wanted = settings.keepAwakeWhileServing
        && (sleepAssertion.isOnACPower || settings.keepAwakeOnBattery)
    } else {
      wanted = false
    }
    sleepAssertion.setActive(wanted)
  }

  /// Called by the settings view when a keep-awake setting changes.
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
    stats: StatsEvent? = nil,
    statsHistory: [StatsSample] = []
  ) -> ServerStore {
    let store = ServerStore(settings: AppSettings.preview(), logs: LogBuffer(), logFile: nil, isLive: false)
    store.runState = runState
    store.info = info
    store.state.apply(.link(link))
    store.state.apply(.engine(engine))
    // One sample of `stats` unless a history is given, which ends with its own.
    for sample in statsHistory.isEmpty ? stats.map { [StatsSample(at: Date(), stats: $0)] } ?? [] : statsHistory {
      store.state.apply(.stats(sample.stats), at: sample.at)
    }
    store.startedAt = Date(timeIntervalSinceNow: -3600)
    return store
  }
}
