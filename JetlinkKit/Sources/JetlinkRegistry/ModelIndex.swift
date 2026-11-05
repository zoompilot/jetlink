import Foundation

/// Which big models a jetlink server can run, and which ONNX each one is: the
/// model index, zoompilot/jetlink-model-index. A port of
/// `jetlink/registry/index.py`, so the server lists and finds the models the
/// comma does.
///
/// A big model is a comma openpilot commit (its ref), and what a server runs
/// is the ONNX that commit was built from. Finding that ONNX takes reading
/// comma's trees, its precompiled pkls' weights and its model repo, so the
/// index repo does it once an hour and publishes the answers:
///
///     default the model a server runs with nothing picked
///     refs    every big-model commit ever seen -> {oid, size}: an old pick still resolves
///     onnx    every ONNX comma has published -> {size}: a pick by hash alone is sized
///     models  what to list, one entry per ONNX, in display order, with its folder
///             and the catalog fields a list in sunnypilot's shape needs
///
/// The server keeps the last index that parsed (`Registry.catalog`), so it
/// works offline with what it had, and looks nothing up itself.
public struct ModelIndex: Sendable, Equatable {
  public static let url = "https://raw.githubusercontent.com/zoompilot/jetlink-model-index/main/index/v1.json"
  /// The index's own version: another one describes fields we would misread.
  public static let schema = 1
  public static let timeout: TimeInterval = 10
  /// How long a fetched index is served before `catalog()` fetches again.
  public static let maxAge: TimeInterval = 3600
  /// A listed model with no folder of its own.
  public static let otherFolder = "Other Models"
  /// The runtime fields of a listed model the index has none for (an index
  /// older than them): what sunnypilot's big models are built for. `RUNTIME`.
  public static let runtime = (generation: "12", environment: "development", runner: "tinygrad")
  /// What runs with nothing picked before there is an index, which names its
  /// own (`default`): Cinque Terre V3 Model (September 17, 2026).
  public static let defaultBigModelRef = "bf3e3631b3f91d92a1020a5e0dd4298b93ff4244"

  /// One listed model.
  public struct Model: Sendable, Equatable {
    public let ref: String
    public let oid: String
    public let size: Int64
    /// As the catalog lists it, build date and all.
    public let name: String
    /// Empty when the index has none.
    public let shortName: String
    public let folder: String
    /// ISO-8601, or empty.
    public let buildTime: String
    /// The comma pull request a model listed only as one is.
    public let pr: Int64?
    public let is20Hz: Bool
    /// The runtime it was built for, as sunnypilot's catalog says.
    public let generation: String
    public let environment: String
    public let runner: String
  }

  /// In display order.
  public let models: [Model]
  /// Every ref the index resolves, the listed ones included.
  public let refs: [String: Pointer]
  /// Every ONNX the index knows, by oid.
  public let sizes: [String: Int64]
  /// The listed ref that runs with nothing picked, nil when the index names
  /// none it lists.
  public let `default`: String?
  /// `refs`' keys in the index's order, as the Python dict keeps them, so
  /// "any ref with this oid" is the same ref on both sides.
  let refOrder: [String]

  public func pointer(ref: String) -> Pointer? {
    refs[ref]
  }

  public func size(oid: String) -> Int64? {
    sizes[oid]
  }

  /// The listed model for a ref's ONNX: its own entry, or for a recompile of
  /// a listed model, the newest ref's (each ONNX is listed once).
  public func listed(ref: String) -> Model? {
    refs[ref].flatMap { listed(oid: $0.oid) }
  }

  /// The first listed model with this ONNX.
  public func listed(oid: String) -> Model? {
    models.first { $0.oid == oid }
  }

  /// A ref for this ONNX: the listed model's, else the first ref the index
  /// resolves to it.
  public func ref(oid: String) -> String? {
    listed(oid: oid)?.ref ?? refOrder.first { refs[$0]?.oid == oid }
  }

  /// `parse_index`: the index, or a registry error naming why it cannot be
  /// used. No IO. An entry that is malformed costs that entry and nothing
  /// else; an index of another schema, or one listing nothing, is refused
  /// whole, so a caller keeps the last one that parsed rather than an empty list.
  static func parse(_ data: JSON) throws(RegistryError) -> ModelIndex {
    guard let object = data.object else { throw .registry("the index is not a JSON object") }
    guard isSchema(object["schema"]) else {
      throw .registry("the index is schema \(repr(object["schema"])), not \(schema)")
    }
    var refs: [String: Pointer] = [:]
    var refOrder: [String] = []
    for (ref, value) in object["refs"]?.object ?? JSONObject() {
      guard CacheLayout.isRef(ref), let found = pointer(value) else { continue }
      if refs.updateValue(found, forKey: ref) == nil { refOrder.append(ref) }
    }
    var sizes: [String: Int64] = [:]
    for (oid, value) in object["onnx"]?.object ?? JSONObject() {
      guard CacheLayout.isSHA256(oid), let size = size(value["size"]) else { continue }
      sizes[oid] = size
    }
    var models: [Model] = []
    var listed = Set<String>()
    for entry in object["models"]?.array ?? [] {
      guard let entry = entry.object, let ref = entry["ref"]?.string, CacheLayout.isRef(ref), !listed.contains(ref),
        let found = pointer(.object(entry))
      else { continue }
      listed.insert(ref)
      models.append(
        Model(
          ref: ref, oid: found.oid, size: found.size,
          name: text(entry["name"]).nonEmpty ?? String(ref.prefix(10)),
          shortName: text(entry["short_name"]),
          folder: text(entry["folder"]).nonEmpty ?? otherFolder,
          buildTime: text(entry["build_time"]),
          pr: size(entry["pr"]),
          is20Hz: entry["is_20hz"].flatMap { if case .bool(let flag) = $0 { flag } else { nil } } ?? true,
          generation: text(entry["generation"]).nonEmpty ?? runtime.generation,
          environment: text(entry["environment"]).nonEmpty ?? runtime.environment,
          runner: text(entry["runner"]).nonEmpty ?? runtime.runner))
    }
    guard !models.isEmpty else { throw .registry("the index lists no model") }
    // a listed model is what is shown and run, so its own entry names its ONNX
    for model in models where refs.updateValue(Pointer(oid: model.oid, size: model.size), forKey: model.ref) == nil {
      refOrder.append(model.ref)
    }
    for ref in refOrder {
      guard let found = refs[ref], sizes[found.oid] == nil else { continue }
      sizes[found.oid] = found.size
    }
    let named = object["default"]?.string
    let chosen = named.flatMap { listed.contains($0) ? $0 : nil }
    return ModelIndex(models: models, refs: refs, sizes: sizes, default: chosen, refOrder: refOrder)
  }

  /// `data.get('schema') == 1`, which Python also holds for 1.0 and True.
  private static func isSchema(_ value: JSON?) -> Bool {
    switch value {
    case .int(let number): return number == Int64(schema)
    case .double(let number): return number == Double(schema)
    case .bool(let flag): return flag && schema == 1
    default: return false
    }
  }

  /// `repr()` of a JSON value as Python's message spells it.
  private static func repr(_ value: JSON?) -> String {
    guard let value else { return "None" }
    if case .string(let text) = value { return JSON.pythonRepr(text) }
    return value.pythonString
  }

  /// `_pointer`: an object with a SHA-256 oid and a size.
  private static func pointer(_ value: JSON) -> Pointer? {
    guard let oid = value["oid"]?.string, CacheLayout.isSHA256(oid), let size = size(value["size"]) else { return nil }
    return Pointer(oid: oid, size: size)
  }

  /// `_size`: a JSON integer above zero. A bool is an int in Python, and a
  /// size is not; neither is 1.5 or "1".
  private static func size(_ value: JSON?) -> Int64? {
    guard case .int(let number) = value, number > 0 else { return nil }
    return number
  }

  /// `_text`: a string, else empty.
  private static func text(_ value: JSON?) -> String {
    value?.string ?? ""
  }
}

extension String {
  /// Python's `text or fallback`, for a string.
  fileprivate var nonEmpty: String? { isEmpty ? nil : self }
}
