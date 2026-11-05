import Foundation

// The control protocol: the events ServerController publishes and the
// commands it runs. No socket carries it: the apps and the status page read
// the controller in process. The JSON is what the status page streams to a
// browser, {"event": "<name>", "t": <float seconds>, ...}, and what the
// Android app gets across JNI; it sends commands as {"cmd": "<name>", ...}.

public enum EngineState: String, Codable, Sendable {
  case none, building, loading, ready, failed
}

public enum LinkState: String, Codable, Sendable {
  case waiting, connected, disconnected
}

/// What the status page opens with: the server's release.
public struct HelloEvent: Codable, Sendable, Equatable {
  public let version: String

  public init(version: String) {
    self.version = version
  }
}

public struct ServerEvent: Codable, Sendable, Equatable {
  public let state: String
  public let detail: String
  public let backend: String?
  public let runtimeVersion: String?
  public let device: String?

  public init(state: String, detail: String, backend: String?, runtimeVersion: String?, device: String?) {
    self.state = state
    self.detail = detail
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
  }
}

public struct LinkEvent: Codable, Sendable, Equatable {
  public let state: LinkState
  public let detail: String
  public let peer: String?
  /// How a connected link is carried, as `LinkMedium` names it; nil from a
  /// server older than the field, or before it can tell.
  public let medium: String?
  /// Who on the comma said hello on a connected link: "modeld" for the drive,
  /// "provision" for the parked model fetch. Nil before any hello, as on the
  /// phone's call the comma holds unread while parked.
  public let client: String?

  public init(state: LinkState, detail: String, peer: String?, medium: String? = nil, client: String? = nil) {
    self.state = state
    self.detail = detail
    self.peer = peer
    self.medium = medium
    self.client = client
  }

  public static let waiting = LinkEvent(state: .waiting, detail: "", peer: nil)

  public var linkMedium: LinkMedium? { medium.flatMap(LinkMedium.init(rawValue:)) }

  /// The comma's driving model holds the link, frames or not: the name
  /// jetlink/openpilot/link.py's Link says in its hello.
  public var isDrive: Bool { state == .connected && client == "modeld" }

  /// What a connected link is carried over, for display: the server's word,
  /// or for a server or comma older than the field, a guess from the peer.
  /// The Python server's USB host says "usb", and a peer on the comma's cable
  /// network is a phone's cable: USB of unknown speed. Anything else is TCP.
  public var connectedMedium: LinkMedium? {
    guard state == .connected else { return nil }
    if let linkMedium { return linkMedium }
    guard let peer, !peer.isEmpty, peer != "usb" else { return .usb }
    return LinkMedium(tcpPeer: peer)
  }
}

/// How the comma's link is carried: the USB generation its controller
/// negotiated, USB of unknown speed, or TCP. The comma's hello says which
/// (`Transport.link_info` in Python), since only its end always knows: a
/// phone's cable is TCP over USB. Names and mapping are `Pinned`.
public enum LinkMedium: String, Codable, Sendable, CaseIterable {
  case usb3, usb2, usb1, usb, tcp

  /// The comma's cable network, "192.168.60.", from its end's address.
  public static let cableNetwork = String(Pinned.cableAddress[...Pinned.cableAddress.lastIndex(of: ".")!])

  /// From a speed as Linux names it: super-speed, high-speed and so on.
  public init(usbSpeed: String?) {
    self = usbSpeed.flatMap { Pinned.usbSpeedMedia[$0] }.flatMap(LinkMedium.init(rawValue:)) ?? .usb
  }

  /// A TCP link, from its peer ("host:port") before a hello says more: the
  /// comma's cable address is a phone's USB cable, of unknown speed. The same
  /// rule as the Python transport's `on_the_cable`.
  public init(tcpPeer peer: String) {
    self = peer.hasPrefix(Pinned.cableAddress + ":") ? .usb : .tcp
  }

  /// From a hello's `client.link`, or nil when it names none.
  public init?(link: [String: Any]?) {
    switch link?["kind"] as? String {
    case "usb", "cable": self.init(usbSpeed: link?["usb_speed"] as? String)
    case "tcp": self = .tcp
    default: return nil
    }
  }

  public var title: String {
    switch self {
    case .usb3: "USB 3"
    case .usb2: "USB 2"
    case .usb1: "USB 1"
    case .usb: "USB"
    case .tcp: "TCP"
    }
  }

  /// A big model's request is 409,600 bytes on the wire: around 1 ms on USB
  /// 3, around 10 ms on USB 2, enough to cost frames and bring the comma near
  /// its soft disable.
  public var isSlow: Bool { self == .usb2 || self == .usb1 }

  /// What to do about a slow link, in a sentence; nil when it is fast enough.
  /// A phone's cable carries the frame as TCP over USB networking, whose USB 2
  /// cost is estimated at 5 ms a frame (docs/iphone-app.md, not yet measured);
  /// a host on the vendor interface pays about 10.
  public func advice(cable: Bool) -> String? {
    guard isSlow else { return nil }
    let cost = cable ? "about 5 ms" : "about 10 ms"
    return "\(title) costs \(cost) a frame more than USB 3. Use a USB 3 cable and port."
  }

  /// `advice(cable:)` for a host on the vendor interface, as the Mac shows it.
  public var advice: String? { advice(cable: false) }
}

public struct EngineEvent: Codable, Sendable, Equatable {
  public let state: EngineState
  public let sha256: String?
  public let detail: String
  public let stage: String?
  public let frac: Double
  public let msg: String
  public let loadOnly: Bool
  /// What runs the ready model, as its backend names it when that is more
  /// than its device says: "NPU(Tensor G5)", or "GPU(fp16)" where the NPU
  /// could not take the model. Nil otherwise, and from older servers.
  public let accelerator: String?

  public init(
    state: EngineState, sha256: String?, detail: String, stage: String?, frac: Double, msg: String, loadOnly: Bool, accelerator: String? = nil
  ) {
    self.state = state
    self.sha256 = sha256
    self.detail = detail
    self.stage = stage
    self.frac = frac
    self.msg = msg
    self.loadOnly = loadOnly
    self.accelerator = accelerator
  }

  public static let none = EngineEvent(state: .none, sha256: nil, detail: "", stage: nil, frac: 0, msg: "", loadOnly: false)
}

public struct StatsEvent: Codable, Sendable, Equatable {
  public struct Total: Codable, Sendable, Equatable {
    public let mean: Double
    public let p99: Double
    public let max: Double

    public init(mean: Double, p99: Double, max: Double) {
      self.mean = mean
      self.p99 = p99
      self.max = max
    }
  }

  /// Means that add up to `servedMs.mean`: staging the inputs, the model run,
  /// the rest of the run, and sending the reply.
  public struct Stages: Codable, Sendable, Equatable {
    public let queue: Double
    public let gpu: Double
    public let other: Double
    public let send: Double

    public init(queue: Double, gpu: Double, other: Double, send: Double) {
      self.queue = queue
      self.gpu = gpu
      self.other = other
      self.send = send
    }
  }

  public struct Mean: Codable, Sendable, Equatable {
    public let mean: Double

    public init(mean: Double) {
      self.mean = mean
    }
  }

  public let frames: Int
  public let fps: Double
  /// From a frame's arrival to its reply leaving.
  public let servedMs: Total
  public let stagesMs: Stages
  public let slow: Int
  public let windowS: Double
  /// From a frame's arrival to its reply being ready, without the send: what
  /// `slow` counts against. Nil from a server that does not send it.
  public let totalMs: Total?
  /// The model run, as the backend times it.
  public let gpuMs: Mean?

  public init(
    frames: Int, fps: Double, servedMs: Total, stagesMs: Stages, slow: Int, windowS: Double, totalMs: Total? = nil, gpuMs: Mean? = nil
  ) {
    self.frames = frames
    self.fps = fps
    self.servedMs = servedMs
    self.stagesMs = stagesMs
    self.slow = slow
    self.windowS = windowS
    self.totalMs = totalMs
    self.gpuMs = gpuMs
  }
}

public struct InventoryModel: Codable, Sendable, Equatable, Identifiable {
  public var id: String { sha256 }
  public let sha256: String
  public let bytes: Int64
  public let path: String
  public let name: String?
  public let ref: String?

  public init(sha256: String, bytes: Int64, path: String, name: String?, ref: String?) {
    self.sha256 = sha256
    self.bytes = bytes
    self.path = path
    self.name = name
    self.ref = ref
  }
}

public struct InventoryArtifact: Codable, Sendable, Equatable, Identifiable {
  public var id: String { key }
  public let sha256: String
  public let key: String
  public let path: String
  public let bytes: Int64
  public let backend: String
  public let runtimeVersion: String?
  public let device: String
  public let builtAt: String?
  public let buildSeconds: Double?
  public let checkpoint: String?
  public let current: Bool

  public init(
    sha256: String, key: String, path: String, bytes: Int64, backend: String, runtimeVersion: String?, device: String, builtAt: String?, buildSeconds: Double?,
    checkpoint: String?, current: Bool
  ) {
    self.sha256 = sha256
    self.key = key
    self.path = path
    self.bytes = bytes
    self.backend = backend
    self.runtimeVersion = runtimeVersion
    self.device = device
    self.builtAt = builtAt
    self.buildSeconds = buildSeconds
    self.checkpoint = checkpoint
    self.current = current
  }
}

public struct InventoryDisk: Codable, Sendable, Equatable {
  public let modelsBytes: Int64
  public let enginesBytes: Int64
  public let freeBytes: Int64

  public init(modelsBytes: Int64, enginesBytes: Int64, freeBytes: Int64) {
    self.modelsBytes = modelsBytes
    self.enginesBytes = enginesBytes
    self.freeBytes = freeBytes
  }
}

public struct InventoryEvent: Codable, Sendable, Equatable {
  public let loaded: String?
  public let lastLoaded: String?
  public let models: [InventoryModel]
  public let artifacts: [InventoryArtifact]
  public let disk: InventoryDisk

  public init(loaded: String?, lastLoaded: String?, models: [InventoryModel], artifacts: [InventoryArtifact], disk: InventoryDisk) {
    self.loaded = loaded
    self.lastLoaded = lastLoaded
    self.models = models
    self.artifacts = artifacts
    self.disk = disk
  }
}

public struct CatalogModel: Codable, Sendable, Equatable, Identifiable {
  public var id: String { ref }
  public let name: String
  public let shortName: String
  public let ref: String
  public let buildTime: String
  public let index: Int
  public let sha256: String?
  public let bytes: Int64?
  /// The model index's folder for it ("Master Models", "Other Models"); nil
  /// from a server that predates the index.
  public let folder: String?

  public init(
    name: String, shortName: String, ref: String, buildTime: String, index: Int, sha256: String?, bytes: Int64?, folder: String? = nil
  ) {
    self.name = name
    self.shortName = shortName
    self.ref = ref
    self.buildTime = buildTime
    self.index = index
    self.sha256 = sha256
    self.bytes = bytes
    self.folder = folder
  }
}

public struct CatalogEvent: Codable, Sendable, Equatable {
  public let fetchedAt: Double?
  public let url: String
  public let defaultRef: String
  public let error: String?
  public let models: [CatalogModel]

  public init(fetchedAt: Double?, url: String, defaultRef: String, error: String?, models: [CatalogModel]) {
    self.fetchedAt = fetchedAt
    self.url = url
    self.defaultRef = defaultRef
    self.error = error
    self.models = models
  }
}

public struct DownloadEvent: Codable, Sendable, Equatable {
  public let sha256: String
  public let ref: String?
  public let state: String
  public let frac: Double
  public let bytes: Int64
  public let total: Int64
  public let rateBps: Double
  public let detail: String
  public let source: String?

  public init(sha256: String, ref: String?, state: String, frac: Double, bytes: Int64, total: Int64, rateBps: Double, detail: String, source: String?) {
    self.sha256 = sha256
    self.ref = ref
    self.state = state
    self.frac = frac
    self.bytes = bytes
    self.total = total
    self.rateBps = rateBps
    self.detail = detail
    self.source = source
  }
}

public struct ImportEvent: Codable, Sendable, Equatable {
  public let path: String
  public let state: String
  public let frac: Double
  public let sha256: String?
  public let detail: String

  public init(path: String, state: String, frac: Double, sha256: String?, detail: String) {
    self.path = path
    self.state = state
    self.frac = frac
    self.sha256 = sha256
    self.detail = detail
  }
}

public struct ReplyEvent: Codable, Sendable, Equatable {
  public let id: Int?
  public let ok: Bool
  public let error: String?
  public let extras: [String: JSONValue]

  public init(id: Int?, ok: Bool, error: String?, extras: [String: JSONValue] = [:]) {
    self.id = id
    self.ok = ok
    self.error = error
    self.extras = extras
  }

  private static let reserved: Set<String> = ["event", "t", "id", "ok", "error"]

  public init(from decoder: any Decoder) throws {
    let object = try decoder.singleValueContainer().decode([String: JSONValue].self)
    self.id = object["id"]?.intValue
    self.ok = object["ok"]?.boolValue ?? false
    self.error = object["error"]?.stringValue
    self.extras = object.filter { !ReplyEvent.reserved.contains($0.key) }
  }

  public func encode(to encoder: any Encoder) throws {
    var object = extras
    object["ok"] = .bool(ok)
    object["id"] = id.map { JSONValue.number(Double($0)) }
    object["error"] = error.map { JSONValue.string($0) }
    var container = encoder.singleValueContainer()
    try container.encode(object)
  }
}

// A JSON value of any shape, so reply extras survive decoding without a schema.
public enum JSONValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case null
  indirect case array([JSONValue])
  indirect case object([String: JSONValue])

  public var stringValue: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  public var numberValue: Double? {
    if case .number(let value) = self { return value }
    return nil
  }

  public var intValue: Int? {
    if case .number(let value) = self { return Int(value) }
    return nil
  }

  public var boolValue: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }

  public var isNull: Bool {
    if case .null = self { return true }
    return false
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Double.self) {
      self = .number(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([JSONValue].self) {
      self = .array(value)
    } else if let value = try? container.decode([String: JSONValue].self) {
      self = .object(value)
    } else {
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "unsupported JSON value")
    }
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let value): try container.encode(value)
    case .number(let value): try container.encode(value)
    case .bool(let value): try container.encode(value)
    case .null: try container.encodeNil()
    case .array(let value): try container.encode(value)
    case .object(let value): try container.encode(value)
    }
  }
}

/// mean, p50, p90, p99 and max of one measure over a benchmark, in ms.
public struct BenchmarkStats: Codable, Sendable, Equatable {
  public let mean: Double
  public let p50: Double
  public let p90: Double
  public let p99: Double
  public let max: Double

  public init(mean: Double, p50: Double, p90: Double, p99: Double, max: Double) {
    self.mean = mean
    self.p50 = p50
    self.p90 = p90
    self.p99 = p99
    self.max = max
  }

  public static let empty = BenchmarkStats(mean: 0, p50: 0, p90: 0, p99: 0, max: 0)
}

/// One window of a benchmark, with the device's thermal state as it closed.
public struct BenchmarkWindow: Codable, Sendable, Equatable {
  public let startSecond: Int
  public let frame: BenchmarkStats
  public let thermal: String

  public init(startSecond: Int, frame: BenchmarkStats, thermal: String) {
    self.startSecond = startSecond
    self.frame = frame
    self.thermal = thermal
  }
}

/// A finished benchmark: the loaded engine run at the comma's pace with
/// nothing on the link, so a phone's numbers can be read at home.
public struct BenchmarkReport: Codable, Sendable, Equatable {
  public let sha256: String
  public let device: String
  public let seconds: Double
  public let frames: Int
  /// The server's share of each frame: queues, model, output.
  public let frame: BenchmarkStats
  /// The model alone: the gpu_us the comma is told.
  public let accelerator: BenchmarkStats
  /// The history queues building the model's inputs, and the output read back.
  public let queues: BenchmarkStats
  public let output: BenchmarkStats
  /// How this build was compiled and what ran beside it, for comparing reports.
  public let build: String
  public let over35: Int
  public let over50: Int
  public let windows: [BenchmarkWindow]
  public let thermalAtStart: String
  public let thermalAtEnd: String
  public let cancelled: Bool

  public init(
    sha256: String, device: String, seconds: Double, frames: Int, frame: BenchmarkStats, accelerator: BenchmarkStats, queues: BenchmarkStats,
    output: BenchmarkStats, build: String, over35: Int, over50: Int, windows: [BenchmarkWindow], thermalAtStart: String, thermalAtEnd: String,
    cancelled: Bool
  ) {
    self.sha256 = sha256
    self.device = device
    self.seconds = seconds
    self.frames = frames
    self.frame = frame
    self.accelerator = accelerator
    self.queues = queues
    self.output = output
    self.build = build
    self.over35 = over35
    self.over50 = over50
    self.windows = windows
    self.thermalAtStart = thermalAtStart
    self.thermalAtEnd = thermalAtEnd
    self.cancelled = cancelled
  }

  /// The report as text, to paste into an issue or a note.
  public var text: String {
    func f(_ s: BenchmarkStats) -> String {
      String(format: "mean %.1f  p50 %.1f  p90 %.1f  p99 %.1f  max %.1f ms", s.mean, s.p50, s.p90, s.p99, s.max)
    }
    var lines = [
      "Jetlink benchmark, \(device)",
      "model \(sha256.prefix(16)), \(frames) frames at 20 Hz over \(Int(seconds)) s\(cancelled ? " (stopped early)" : "")",
      build,
      "frame        \(f(frame))",
      "accelerator  \(f(accelerator))",
      "queues       \(f(queues))",
      "output       \(f(output))",
      "over 35 ms: \(over35)   over 50 ms: \(over50)",
      "temperature: \(thermalAtStart) at start, \(thermalAtEnd) at end",
    ]
    if !windows.isEmpty {
      lines.append("by window:")
      for w in windows {
        lines.append(String(format: "  %4d s  mean %5.1f  p99 %5.1f  max %5.1f ms  ", w.startSecond, w.frame.mean, w.frame.p99, w.frame.max) + w.thermal)
      }
    }
    return lines.joined(separator: "\n")
  }
}

/// A benchmark's progress, and at its end the report. `state` is running,
/// done, cancelled or failed; `frame` is the running frame stats so far.
public struct BenchmarkEvent: Codable, Sendable, Equatable {
  public let state: String
  public let elapsed: Double
  public let total: Double
  public let frames: Int
  public let frame: BenchmarkStats?
  public let report: BenchmarkReport?
  public let detail: String

  public init(state: String, elapsed: Double, total: Double, frames: Int, frame: BenchmarkStats?, report: BenchmarkReport?, detail: String) {
    self.state = state
    self.elapsed = elapsed
    self.total = total
    self.frames = frames
    self.frame = frame
    self.report = report
    self.detail = detail
  }

  public var isFinished: Bool { state != "running" }
}

/// The comma asked the server to power its device off. The Swift server
/// answers no (a phone does not power itself off for the comma) and tells
/// the app, which tells the person.
public struct ShutdownRequestEvent: Codable, Sendable, Equatable {
  public let reason: String

  public init(reason: String) {
    self.reason = reason
  }
}

public enum ControlEvent: Sendable, Equatable {
  case hello(HelloEvent)
  case server(ServerEvent)
  case link(LinkEvent)
  case engine(EngineEvent)
  case stats(StatsEvent)
  case inventory(InventoryEvent)
  case catalog(CatalogEvent)
  case download(DownloadEvent)
  case importEvent(ImportEvent)
  case benchmark(BenchmarkEvent)
  case shutdownRequest(ShutdownRequestEvent)
  case reply(ReplyEvent)

  /// The event's name on the wire: "hello", "import", "shutdown_request".
  public var name: String {
    switch self {
    case .hello: "hello"
    case .server: "server"
    case .link: "link"
    case .engine: "engine"
    case .stats: "stats"
    case .inventory: "inventory"
    case .catalog: "catalog"
    case .download: "download"
    case .importEvent: "import"
    case .benchmark: "benchmark"
    case .shutdownRequest: "shutdown_request"
    case .reply: "reply"
    }
  }

  /// The event as the status page streams it: one JSON object with `event`,
  /// `t` (Unix seconds) and the payload's keys, then a newline.
  public func jsonLine(at date: Date = Date()) -> Data {
    ControlJSON.line(event: name, payload(), at: date) ?? Data("{}\n".utf8)
  }

  /// The payload alone, as JSONSerialization objects.
  public func payload() -> [String: Any] {
    (Mirror(reflecting: self).children.first?.value as? any Encodable).map { ControlEvent.object($0) } ?? [:]
  }

  /// A value of the protocol as its JSON object: snake_case keys, and no key
  /// for an absent optional, which every reader takes as null.
  static func object(_ value: some Encodable) -> [String: Any] {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard let data = try? encoder.encode(value) else { return [:] }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
  }
}

/// JSON as every jetlink reader gets it: keys sorted, slashes left alone.
public enum ControlJSON {
  /// Nil for what JSON cannot hold.
  public static func data(_ object: Any) -> Data? {
    guard JSONSerialization.isValidJSONObject(object) else { return nil }
    return try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
  }

  /// One event: `fields` with `event` and `t` (Unix seconds), then a newline.
  public static func line(event: String, _ fields: [String: Any], at date: Date) -> Data? {
    var object = fields
    object["event"] = event
    object["t"] = date.timeIntervalSince1970
    return data(object).map { $0 + Data("\n".utf8) }
  }
}

public enum ControlCommand: Sendable, Equatable {
  case status
  case catalog(refresh: Bool)
  case download(ref: String?, sha256: String?)
  case cancelDownload(sha256: String)
  case importModel(path: String)
  /// Use this model here: a pick the comma takes at its next hello. `ref`
  /// is the catalog row chosen, for a model two rows share.
  case prepare(sha256: String, frameSkip: Int, ref: String? = nil)
  case unload
  case forget(sha256: String, artifacts: Bool, model: Bool)
  case inventory
  case shutdown
  /// Run the loaded engine at the comma's pace for `seconds`, with no comma.
  case benchmark(seconds: Double)
  case cancelBenchmark

  public var name: String {
    switch self {
    case .status: return "status"
    case .catalog: return "catalog"
    case .download: return "download"
    case .cancelDownload: return "cancel_download"
    case .importModel: return "import"
    case .prepare: return "prepare"
    case .unload: return "unload"
    case .forget: return "forget"
    case .inventory: return "inventory"
    case .shutdown: return "shutdown"
    case .benchmark: return "benchmark"
    case .cancelBenchmark: return "cancel_benchmark"
    }
  }

  /// The command an object names, as the Android app sends it. Absent
  /// arguments take the Python server's defaults.
  public init(object: [String: Any]) throws {
    let name = object["cmd"] as? String ?? ""
    func text(_ key: String) throws -> String {
      guard let value = object[key] as? String, !value.isEmpty else { throw Invalid(description: "\(name) needs \(key)") }
      return value
    }
    func flag(_ key: String, _ fallback: Bool) -> Bool { (object[key] as? Bool) ?? fallback }
    func number(_ key: String) -> NSNumber? { object[key] as? NSNumber }
    switch name {
    case "status": self = .status
    case "catalog": self = .catalog(refresh: flag("refresh", false))
    case "download": self = .download(ref: object["ref"] as? String, sha256: object["sha256"] as? String)
    case "cancel_download": self = .cancelDownload(sha256: try text("sha256"))
    case "import": self = .importModel(path: try text("path"))
    case "prepare":
      self = .prepare(
        sha256: try text("sha256"), frameSkip: number("frame_skip")?.intValue ?? Pinned.defaultFrameSkip, ref: object["ref"] as? String)
    case "unload": self = .unload
    case "forget": self = .forget(sha256: try text("sha256"), artifacts: flag("artifacts", true), model: flag("model", false))
    case "inventory": self = .inventory
    case "shutdown": self = .shutdown
    case "benchmark": self = .benchmark(seconds: number("seconds")?.doubleValue ?? 60)
    case "cancel_benchmark": self = .cancelBenchmark
    default: throw Invalid(description: "unknown command \(name)")
    }
  }

  /// A command object the server cannot run.
  public struct Invalid: Error, CustomStringConvertible {
    public let description: String
  }
}
