import Foundation
import Observation

/// What an app shows of its server, kept from the server's events. The
/// iPhone's PhoneServer, the Mac's ServerStore and Android's AppSnapshot
/// each keep one, so an event means the same on every screen. Observable
/// property by property: a view that shows the link is not redrawn for
/// every stats event.
@Observable
public final class ServerViewState {
  /// backend, runtime version and device, as the hello reports them.
  public private(set) var server: ServerEvent?
  public private(set) var link: LinkEvent = .waiting
  public private(set) var engine: EngineEvent = .none
  /// The last two minutes of `stats`, oldest first, while a comma stays connected.
  public private(set) var statsHistory: [StatsSample] = []
  /// The benchmark running or last run, if any.
  public private(set) var benchmark: BenchmarkEvent?
  /// The comma's last request to power the server's device off, which it
  /// refused, and how many there have been, so each one is shown.
  public private(set) var shutdownRequest: ShutdownRequestEvent?
  public private(set) var shutdownRequests = 0

  public init() {}

  /// The latest `stats`.
  public var stats: StatsEvent? { statsHistory.last?.stats }

  /// Folds `event` in, `date` being when it arrived. False for what this
  /// leaves to others: the model events (inventory, catalog, download,
  /// import), which a ModelStore keeps, and what no screen shows.
  @discardableResult
  public func apply(_ event: ControlEvent, at date: Date = Date()) -> Bool {
    switch event {
    case .server(let value):
      server = value
    case .link(let value):
      link = value
      // the numbers were the comma's that went
      if value.state != .connected { statsHistory = [] }
    case .engine(let value):
      engine = value
    case .stats(let value):
      statsHistory = StatsSample.appending(value, at: date, to: statsHistory)
    case .benchmark(let value):
      benchmark = value
    case .shutdownRequest(let value):
      shutdownRequest = value
      shutdownRequests += 1
    case .inventory, .catalog, .download, .importEvent, .hello, .reply:
      return false
    }
    return true
  }

  /// The server stopped: no comma, no engine, no numbers. The last
  /// benchmark stays to be read.
  public func serverStopped() {
    link = .waiting
    engine = .none
    statsHistory = []
    // A benchmark ends with its server; a finished one's report stays.
    if benchmark?.isFinished == false { benchmark = nil }
  }

  /// A new server: nothing from the last one. The shutdown requests keep
  /// counting, so one after a restart is still news.
  public func reset() {
    serverStopped()
    server = nil
    benchmark = nil
    shutdownRequest = nil
  }
}
