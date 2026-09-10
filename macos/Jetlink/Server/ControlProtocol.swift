import Foundation

// The control protocol, version 1. One JSON object per line, UTF-8, newline terminated.
// Client to server: {"id": <int>, "cmd": "<name>", ...arguments}
// Server to client: {"event": "<name>", "t": <float seconds>, ...}

enum EngineState: String, Codable, Sendable {
  case none, building, loading, ready, failed
}

enum LinkState: String, Codable, Sendable {
  case waiting, connected, disconnected
}

struct HelloEvent: Codable, Sendable, Equatable {
  let protocolVersion: Int
  let pid: Int32
  let version: String
  let python: String
  let platform: String
  let cache: String
  let transport: String
  let port: Int?

  // "protocol" carries no underscore, so the snake case strategy leaves it alone.
  enum CodingKeys: String, CodingKey {
    case protocolVersion = "protocol"
    case pid, version, python, platform, cache, transport, port
  }
}

struct ServerEvent: Codable, Sendable, Equatable {
  let state: String
  let detail: String
  let backend: String?
  let runtimeVersion: String?
  let device: String?
}

struct LinkEvent: Codable, Sendable, Equatable {
  let state: LinkState
  let detail: String
  let peer: String?

  init(state: LinkState, detail: String, peer: String?) {
    self.state = state
    self.detail = detail
    self.peer = peer
  }

  static let waiting = LinkEvent(state: .waiting, detail: "", peer: nil)
}

struct EngineEvent: Codable, Sendable, Equatable {
  let state: EngineState
  let sha256: String?
  let detail: String
  let stage: String?
  let frac: Double
  let msg: String
  let loadOnly: Bool

  init(state: EngineState, sha256: String?, detail: String, stage: String?, frac: Double, msg: String, loadOnly: Bool) {
    self.state = state
    self.sha256 = sha256
    self.detail = detail
    self.stage = stage
    self.frac = frac
    self.msg = msg
    self.loadOnly = loadOnly
  }

  static let none = EngineEvent(state: .none, sha256: nil, detail: "", stage: nil, frac: 0, msg: "", loadOnly: false)
}

struct StatsEvent: Codable, Sendable, Equatable {
  struct Total: Codable, Sendable, Equatable {
    let mean: Double
    let p99: Double
    let max: Double
  }

  struct Gpu: Codable, Sendable, Equatable {
    let mean: Double
  }

  let frames: Int
  let fps: Double
  let totalMs: Total
  let gpuMs: Gpu
  let slow: Int
  let windowS: Double
}

struct InventoryModel: Codable, Sendable, Equatable, Identifiable {
  var id: String { sha256 }
  let sha256: String
  let bytes: Int64
  let path: String
  let name: String?
  let ref: String?
}

struct InventoryArtifact: Codable, Sendable, Equatable, Identifiable {
  var id: String { key }
  let sha256: String
  let key: String
  let path: String
  let bytes: Int64
  let backend: String
  let runtimeVersion: String?
  let device: String
  let builtAt: String?
  let buildSeconds: Double?
  let checkpoint: String?
  let current: Bool
}

struct InventoryDisk: Codable, Sendable, Equatable {
  let modelsBytes: Int64
  let enginesBytes: Int64
  let freeBytes: Int64
}

struct InventoryEvent: Codable, Sendable, Equatable {
  let loaded: String?
  let lastLoaded: String?
  let models: [InventoryModel]
  let artifacts: [InventoryArtifact]
  let disk: InventoryDisk
}

struct CatalogModel: Codable, Sendable, Equatable, Identifiable {
  var id: String { ref }
  let name: String
  let shortName: String
  let ref: String
  let buildTime: String
  let index: Int
  let sha256: String?
  let bytes: Int64?
}

struct CatalogEvent: Codable, Sendable, Equatable {
  let fetchedAt: Double?
  let url: String
  let defaultRef: String
  let error: String?
  let models: [CatalogModel]
}

struct DownloadEvent: Codable, Sendable, Equatable {
  let sha256: String
  let ref: String?
  let state: String
  let frac: Double
  let bytes: Int64
  let total: Int64
  let rateBps: Double
  let detail: String
  let source: String?
}

struct ImportEvent: Codable, Sendable, Equatable {
  let path: String
  let state: String
  let frac: Double
  let sha256: String?
  let detail: String
}

struct ReplyEvent: Codable, Sendable, Equatable {
  let id: Int?
  let ok: Bool
  let error: String?
  let extras: [String: JSONValue]

  init(id: Int?, ok: Bool, error: String?, extras: [String: JSONValue] = [:]) {
    self.id = id
    self.ok = ok
    self.error = error
    self.extras = extras
  }

  private static let reserved: Set<String> = ["event", "t", "id", "ok", "error"]

  init(from decoder: any Decoder) throws {
    let object = try decoder.singleValueContainer().decode([String: JSONValue].self)
    self.id = object["id"]?.intValue
    self.ok = object["ok"]?.boolValue ?? false
    self.error = object["error"]?.stringValue
    self.extras = object.filter { !ReplyEvent.reserved.contains($0.key) }
  }

  func encode(to encoder: any Encoder) throws {
    var object = extras
    object["ok"] = .bool(ok)
    object["id"] = id.map { JSONValue.number(Double($0)) } ?? .null
    object["error"] = error.map { JSONValue.string($0) } ?? .null
    var container = encoder.singleValueContainer()
    try container.encode(object)
  }
}

// A JSON value of any shape, so reply extras survive decoding without a schema.
enum JSONValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case null
  indirect case array([JSONValue])
  indirect case object([String: JSONValue])

  var stringValue: String? {
    if case .string(let value) = self { return value }
    return nil
  }

  var numberValue: Double? {
    if case .number(let value) = self { return value }
    return nil
  }

  var intValue: Int? {
    if case .number(let value) = self { return Int(value) }
    return nil
  }

  var boolValue: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }

  var isNull: Bool {
    if case .null = self { return true }
    return false
  }

  init(from decoder: any Decoder) throws {
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

  func encode(to encoder: any Encoder) throws {
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

enum ControlEvent: Sendable, Equatable {
  case hello(HelloEvent)
  case server(ServerEvent)
  case link(LinkEvent)
  case engine(EngineEvent)
  case stats(StatsEvent)
  case inventory(InventoryEvent)
  case catalog(CatalogEvent)
  case download(DownloadEvent)
  case importEvent(ImportEvent)
  case reply(ReplyEvent)
  case unknown(name: String)

  private struct Envelope: Decodable {
    let event: String
  }

  static func makeDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
  }

  init(jsonLine: Data) throws {
    let decoder = ControlEvent.makeDecoder()
    let name = try decoder.decode(Envelope.self, from: jsonLine).event
    switch name {
    case "hello": self = .hello(try decoder.decode(HelloEvent.self, from: jsonLine))
    case "server": self = .server(try decoder.decode(ServerEvent.self, from: jsonLine))
    case "link": self = .link(try decoder.decode(LinkEvent.self, from: jsonLine))
    case "engine": self = .engine(try decoder.decode(EngineEvent.self, from: jsonLine))
    case "stats": self = .stats(try decoder.decode(StatsEvent.self, from: jsonLine))
    case "inventory": self = .inventory(try decoder.decode(InventoryEvent.self, from: jsonLine))
    case "catalog": self = .catalog(try decoder.decode(CatalogEvent.self, from: jsonLine))
    case "download": self = .download(try decoder.decode(DownloadEvent.self, from: jsonLine))
    case "import": self = .importEvent(try decoder.decode(ImportEvent.self, from: jsonLine))
    case "reply": self = .reply(try decoder.decode(ReplyEvent.self, from: jsonLine))
    default: self = .unknown(name: name)
    }
  }

  var replyEvent: ReplyEvent? {
    if case .reply(let reply) = self { return reply }
    return nil
  }
}

enum ControlCommand: Sendable, Equatable {
  case status
  case catalog(refresh: Bool)
  case download(ref: String?, sha256: String?)
  case cancelDownload(sha256: String)
  case importModel(path: String)
  case prepare(sha256: String, frameSkip: Int)
  case unload
  case forget(sha256: String, artifacts: Bool, model: Bool)
  case inventory
  case shutdown

  var name: String {
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
    }
  }

  // Absent optional arguments are written as null, never left out, the way
  // section 4 of the contract describes the whole protocol.
  private var arguments: [String: Any] {
    switch self {
    case .status, .unload, .inventory, .shutdown:
      return [:]
    case .catalog(let refresh):
      return ["refresh": refresh]
    case .download(let ref, let sha256):
      return ["ref": ref ?? NSNull(), "sha256": sha256 ?? NSNull()]
    case .cancelDownload(let sha256):
      return ["sha256": sha256]
    case .importModel(let path):
      return ["path": path]
    case .prepare(let sha256, let frameSkip):
      return ["sha256": sha256, "frame_skip": frameSkip]
    case .forget(let sha256, let artifacts, let model):
      return ["sha256": sha256, "artifacts": artifacts, "model": model]
    }
  }

  func jsonLine(id: Int) -> Data {
    var object: [String: Any] = arguments
    object["id"] = id
    object["cmd"] = name
    guard var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]) else {
      return Data("{}\n".utf8)
    }
    data.append(0x0A)
    return data
  }
}
