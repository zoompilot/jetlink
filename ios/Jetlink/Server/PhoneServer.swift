import Foundation
import JetlinkKit
import JetlinkORT
import JetlinkRegistry
import JetlinkServer
import Observation
import os

/// The jetlink server inside the app, and everything the views read about it.
/// The iPhone's `ServerStore`: the same events and commands the Mac app has,
/// with the server in process instead of a Python child behind a socket.
@MainActor
@Observable
final class PhoneServer: ServerControlling {
  private(set) var runState: ServerRunState = .stopped
  /// What the screens show of the server, kept from its events.
  let state = ServerViewState()
  var link: LinkEvent { state.link }
  var engine: EngineEvent { state.engine }
  private(set) var port: UInt16?
  /// The last ten seconds of frames, refreshed once a second while serving.
  private(set) var recent: StatsEvent?
  private(set) var lastFailure: String?
  let modelEvents: AsyncStream<ControlEvent>
  /// The profile the running server was started on.
  private(set) var profile = Processor.fallback
  /// A processor comparison runs each candidate in turn: while this is set
  /// the server runs it, whatever the setting says (`restart(on:)`).
  @ObservationIgnored private var profileOverride: OrtProfile?
  /// A model on its way to a server on its own profile (`followModel`).
  @ObservationIgnored private var following: String?

  /// How far back the dashboard's headline reaches.
  static let recentWindow: TimeInterval = 10

  let settings: PhoneSettings
  /// Where the cable is: the server dials the comma while the phone has an
  /// address on the comma's network.
  let network: NetworkInterfaces
  @ObservationIgnored private let modelEventsContinuation: AsyncStream<ControlEvent>.Continuation
  @ObservationIgnored private var embedded: EmbeddedServer?
  private var server: Server? { embedded?.server }
  @ObservationIgnored private var consumeTask: Task<Void, Never>?
  @ObservationIgnored private var recentTask: Task<Void, Never>?
  @ObservationIgnored private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "app")

  /// What the in-process server logs, for a Logs view.
  let logs = LogBuffer()

  init(settings: PhoneSettings, network: NetworkInterfaces) {
    self.settings = settings
    self.network = network
    (modelEvents, modelEventsContinuation) = AsyncStream.makeStream(bufferingPolicy: .unbounded)
    let logs = self.logs
    let stream = LogStream(format: { PhoneServer.logLine($0, $1, $2) })
    Task {
      for await line in stream.lines {
        logs.append(line)
      }
    }
    watchCable()
  }

  nonisolated private static let clock = Date.ISO8601FormatStyle().time(includingFractionalSeconds: true)

  /// One line as the sink writes them: time, level, category, message.
  nonisolated static func logLine(_ level: Log.Level, _ category: String, _ message: String) -> String {
    "\(Date().formatted(clock)) \(level.rawValue.uppercased()) \(category): \(message)"
  }

  /// A line from the app itself, beside the server's, so a Logs view has
  /// the phone's story in one place.
  func note(_ level: Log.Level, _ message: String) {
    switch level {
    case .debug: log.debug("\(message, privacy: .public)")
    case .info: log.info("\(message, privacy: .public)")
    case .warning: log.warning("\(message, privacy: .public)")
    case .error: log.error("\(message, privacy: .public)")
    }
    logs.append(PhoneServer.logLine(level, "app", message))
  }

  // MARK: the cable

  /// Over the cable the comma listens and the phone dials.
  static let commaDial = DialTarget(host: NetworkInterfaces.commaAddress, port: Wire.defaultPort)

  /// What the connected comma's link is carried over: USB 3, USB 2 or TCP.
  var linkMedium: LinkMedium? { link.connectedMedium }

  /// Follows the phone's addresses: the comma is dialed while the cable's
  /// lease is there, and left alone once it is gone.
  private func watchCable() {
    withObservationTracking {
      updateDial()
    } onChange: { [weak self] in
      Task { @MainActor in self?.watchCable() }
    }
  }

  private func updateDial() {
    let target = network.cable == nil ? nil : PhoneServer.commaDial
    server?.setDial(target)
  }

  // MARK: lifecycle

  func start() {
    switch runState {
    case .stopped, .failed: break
    default: return
    }
    runState = .starting
    lastFailure = nil
    let profile = profileOverride ?? following.map(settings.profile(for:)) ?? wantedProfile
    self.profile = profile
    do {
      let root = try PhoneServer.prepareCacheDirectory()
      let embedded = try EmbeddedServer(
        // The listener is for a comma on Wi-Fi and bench tools; over the cable the phone dials the comma.
        configuration: Server.Configuration(port: settings.port, cacheRoot: root, preload: following == nil, listen: settings.listening, keepPlans: 2),
        backend: OrtBackend(
          profile: profile, preparer: ONNXPreparer(), keepAlive: settings.keepGPUAwake, keepCPUWarm: settings.keepCPUWarm))
      self.embedded = embedded
      consumeTask = Task { [weak self] in
        for await event in embedded.events {
          self?.apply(event)
        }
      }
      try embedded.start()
      let server = embedded.server
      port = server.port
      runState = .serving
      startRecentTicker()
      updateDial()
      log.info("serving on \(profile.rawValue)\(server.port.map { ", port \($0)" } ?? ", not listening")")
    } catch {
      fail("Jetlink could not start its server: \(error)")
    }
  }

  /// Stops serving. The engine stays loaded in the server until it is let
  /// go; `shutdown()` is for the app's end.
  func stop() {
    tearDown(release: false)
  }

  /// The app is terminating: stop, and release the engine with it.
  func shutdown() {
    tearDown(release: true)
  }

  private func tearDown(release: Bool) {
    recentTask?.cancel()
    recentTask = nil
    embedded?.stop(releasingEngine: release)
    consumeTask?.cancel()
    consumeTask = nil
    embedded = nil
    runState = .stopped
    resetLiveState()
  }

  /// A change of port, listening or compute device takes a new server; the engine is
  /// loaded again, or prepared again for a new device.
  func restart() {
    stop()
    start()
  }

  /// The profile the setting asks for: Automatic's goes by the model the
  /// server loads at its start, the one it had loaded last.
  private var wantedProfile: OrtProfile {
    settings.profile(for: CacheLayout(root: PhoneSettings.cacheDirectory).lastLoaded()?.sha256)
  }

  /// A server on `profile` whatever the setting says, for a processor
  /// comparison; nil goes back to the setting, with no restart when the
  /// server is on its profile already.
  func restart(on profile: OrtProfile?) {
    profileOverride = profile
    if profile == nil, runState == .serving, self.profile == wantedProfile { return }
    restart()
  }

  /// iOS takes a suspended app's listening socket; listen again on the way back.
  func becameActive() {
    guard runState == .serving, let server, !server.isListening else { return }
    do {
      try server.reopenListener()
      log.info("listening again after the app was suspended")
    } catch {
      fail("Jetlink could not listen again after being in the background: \(error)")
    }
  }

  /// The card on Status says it failed; the reason is in Logs.
  private func fail(_ detail: String) {
    note(.error, detail)
    lastFailure = detail
    runState = .failed(detail)
    embedded?.stop(releasingEngine: false)
    embedded = nil
    resetLiveState()
  }

  private static func prepareCacheDirectory() throws -> URL {
    var root = PhoneSettings.cacheDirectory
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    // Gigabytes of models and engines have no place in an iCloud backup.
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try? root.setResourceValues(values)
    return root
  }

  private func startRecentTicker() {
    recentTask?.cancel()
    recentTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard let self, let server = self.server else { return }
        self.recent = self.link.state == .connected ? server.recentStats(window: PhoneServer.recentWindow) : nil
      }
    }
  }

  // MARK: ServerControlling

  func send(_ command: ControlCommand) async throws -> ReplyEvent {
    guard let embedded else { throw ServerUnavailable() }
    return await embedded.handle(command)
  }

  /// The loaded engine's outputs on the comparison's fixed frames.
  func probe() async throws -> ProbeCapture {
    guard let host = server?.host else { throw ServerUnavailable() }
    return try await Task.detached { try host.probe() }.value
  }

  func startIfNeeded() async throws {
    if runState != .serving {
      start()
    }
    if case .failed(let detail) = runState {
      throw ServerUnavailable(detail: detail)
    }
  }

  struct ServerUnavailable: LocalizedError {
    var detail = "The server is not running."
    var errorDescription: String? { detail }
  }

  // MARK: events

  func apply(_ event: ControlEvent) {
    if !state.apply(event) {
      modelEventsContinuation.yield(event)
    }
    if case .link(let value) = event, value.state != .connected {
      recent = nil
    }
    if case .engine(let value) = event, let sha256 = value.sha256 {
      followModel(sha256)
    }
  }

  /// Automatic runs each model on what a comparison picked for it, and a
  /// server runs one profile: a model with another pick takes a server on
  /// that profile, which is asked for the model in place of the one it
  /// would load at its start.
  private func followModel(_ sha256: String) {
    if following == sha256 {
      following = nil
    }
    guard following == nil, profileOverride == nil, runState == .serving else { return }
    let wanted = settings.profile(for: sha256)
    guard wanted != profile else { return }
    note(.info, "this model runs on \(wanted.rawValue) here; restarting the server from \(profile.rawValue)")
    following = sha256
    restart()
    Task {
      let reply = try? await send(.prepare(sha256: sha256, frameSkip: ModelStore.defaultFrameSkip))
      if reply?.ok != true, following == sha256 {
        following = nil
      }
    }
  }

  private func resetLiveState() {
    state.serverStopped()
    recent = nil
    port = nil
  }
}
