import Foundation
import JetlinkKit
import JetlinkLog
import JetlinkRegistry

/// The jetlink server in process, as an app runs it: the server, its control
/// semantics, and the model registry on the same cache. The iPhone's
/// `PhoneServer` and the Mac's `ServerStore` each drive one, so both apps run
/// the same code between their views and the comma.
public final class EmbeddedServer: @unchecked Sendable {
  public let server: Server
  public let controller: ServerController
  /// The comma's add-on models, served beside the driving model (docs/addons.md).
  public let addons: AddonService?

  /// The control channel's events, as the Python server's socket sends them.
  public var events: AsyncStream<ControlEvent> { controller.events }

  /// The server on `configuration` and the host's `backend`, `gadget` and
  /// `hooks`, creating its cache directory. With `addonSource`, the comma's
  /// add-on interfaces are served too, their models built on `addonBackend`
  /// (the host's backend without one).
  public convenience init(
    configuration: Server.Configuration, backend: any EngineBackend, gadget: (any GadgetSource)? = nil, hooks: ServerHooks = ServerHooks(),
    addonBackend: (any EngineBackend)? = nil, addonSource: (@Sendable ([UInt8]) -> any AddonSource)? = nil
  ) throws {
    try FileManager.default.createDirectory(at: configuration.cacheRoot, withIntermediateDirectories: true)
    try self.init(
      server: Server(configuration: configuration, backend: backend, gadget: gadget, hooks: hooks), addonBackend: addonBackend, addonSource: addonSource)
  }

  private init(server: Server, addonBackend: (any EngineBackend)?, addonSource: (@Sendable ([UInt8]) -> any AddonSource)?) {
    self.server = server
    controller = ServerController(server: server, registry: Registry(layout: CacheLayout(root: server.configuration.cacheRoot)))
    addons = addonSource.map {
      AddonService(host: server.host, backend: addonBackend, cacheRoot: server.configuration.cacheRoot, source: $0)
    }
  }

  /// Serves, and publishes the state a client's first screen needs.
  /// Subscribe to `events` first: nothing published before is kept.
  public func start() throws {
    addons?.attach()
    try server.start()
    addons?.start()
    controller.publishInitialState()
  }

  public func handle(_ command: ControlCommand) async -> ReplyEvent {
    await controller.handle(command)
  }

  /// Stops serving and ends `events`. With `releasingEngine` the engine is
  /// let go at once, as when the app ends; without it the engine goes with
  /// the last reference to this server.
  public func stop(releasingEngine: Bool) {
    controller.finish()
    addons?.stop()
    if releasingEngine {
      server.shutdown()
    } else {
      server.stop()
    }
  }

  /// A log line as the Python server writes them,
  /// "%(asctime)s %(levelname)-7s %(name)s: %(message)s", so one Logs view
  /// and one log file read the same whichever server wrote them.
  public static func logLine(_ level: Log.Level, _ category: String, _ message: String, at date: Date = Date()) -> String {
    "\(logTimestamp(date, separator: ",")) \(level.name.padding(toLength: 7, withPad: " ", startingAt: 0)) jetlink.\(category): \(message)"
  }
}

/// The server's log lines as one ordered stream. The sink only formats and
/// yields; one consumer appends to a Logs view or a file. A task per line
/// hopped to the main actor for each line and could reorder them.
public final class LogStream: Sendable {
  public let lines: AsyncStream<String>
  private let continuation: AsyncStream<String>.Continuation

  /// Installs `Log.sink`, formatting each line with `format`.
  public init(format: @escaping @Sendable (Log.Level, String, String) -> String = { EmbeddedServer.logLine($0, $1, $2) }) {
    let (lines, continuation) = AsyncStream<String>.makeStream(bufferingPolicy: .bufferingNewest(5000))
    self.lines = lines
    self.continuation = continuation
    Log.sink = { level, category, message in continuation.yield(format(level, category, message)) }
  }

  /// Removes the sink and ends `lines`.
  public func finish() {
    Log.sink = nil
    continuation.finish()
  }
}
