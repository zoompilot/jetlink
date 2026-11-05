import Foundation

/// Where built engines, models and the registry's state live. The layout is
/// the Python server's (`jetlink/server/cache.py`), so a cache means the same
/// thing to either:
///
///     <root>/engines/<sha16>.<backend tag><suffix>     the artifact
///     <root>/engines/<sha16>.<backend tag>.json        its sidecar: build facts and the model spec
///     <root>/models/<sha16>.onnx                       the model as uploaded, never pruned
///     <root>/registry/                                 index.json, local-models.json
///     <root>/last-loaded.json                          what the server loaded last
public struct CacheLayout: Sendable, Equatable {
  /// Artifacts of one backend kept by `EngineCache.prune`. One per listed
  /// model or so: a rebuild costs minutes, so a smaller cap rebuilds on every
  /// A/B switch.
  public static let keepArtifacts = 6

  public let root: URL

  public init(root: URL) {
    self.root = root
  }

  public var models: URL { root.appending(path: "models", directoryHint: .isDirectory) }
  public var engines: URL { root.appending(path: "engines", directoryHint: .isDirectory) }
  public var registry: URL { root.appending(path: "registry", directoryHint: .isDirectory) }

  /// The last model index that parsed, as it was served.
  public var indexURL: URL { registry.appending(path: "index.json") }
  public var localModelsURL: URL { registry.appending(path: "local-models.json") }

  /// Beside the caches rather than in them: it describes the server, not an
  /// artifact.
  public var lastLoadedURL: URL { root.appending(path: "last-loaded.json") }
  /// The big model picked last, here or on the comma (JetlinkServer's PickStore).
  public var pickURL: URL { root.appending(path: "pick.json") }

  /// Where the server looks for a model's ONNX. The identity arrives from the
  /// peer and becomes a path, so anything but a digest is refused.
  public func modelPath(sha256: String) throws(RegistryError) -> URL {
    try CacheLayout.validate(sha256: sha256)
    return models.appending(path: "\(sha256.prefix(16)).onnx")
  }

  /// A model identity: 64 lowercase hex characters, which is also the LFS oid.
  public static func isSHA256(_ value: String) -> Bool {
    isLowercaseHex(value, count: 64)
  }

  /// A comma openpilot commit: 40 lowercase hex characters.
  public static func isRef(_ value: String) -> Bool {
    isLowercaseHex(value, count: 40)
  }

  public static func validate(sha256: String) throws(RegistryError) {
    guard isSHA256(sha256) else { throw .invalidIdentity }
  }

  static func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
  }

  /// What the server loaded last, for a fresh process to preload. A marker
  /// written before there were backends has no backend field and still reads.
  public func lastLoaded() -> LastLoaded? {
    guard let marker = Files.readJSON(lastLoadedURL),
      let sha256 = marker["sha256"]?.string,
      let frameSkip = marker["frame_skip"]?.pythonInt,
      CacheLayout.isSHA256(sha256)
    else { return nil }
    return LastLoaded(sha256: sha256, frameSkip: Int(frameSkip))
  }

  /// Drops build directories a crashed or killed build left behind. Builds
  /// stage their artifact in a `tmp*` directory inside engines/, which prune
  /// does not look at.
  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    let now = Date().timeIntervalSince1970
    for name in Files.names(in: engines) where name.hasPrefix("tmp") {
      let url = engines.appending(path: name)
      guard let status = Files.status(url), status.isDirectory, now - status.modified >= maxAge else { continue }
      try? FileManager.default.removeItem(at: url)
    }
  }

  /// The registry and models directories, which the Python registry makes
  /// when it opens a cache.
  func makeRegistryDirectories() {
    try? Files.makeDirectory(registry)
    try? Files.makeDirectory(models)
  }
}

public struct LastLoaded: Sendable, Equatable {
  public let sha256: String
  public let frameSkip: Int

  public init(sha256: String, frameSkip: Int) {
    self.sha256 = sha256
    self.frameSkip = frameSkip
  }
}

/// One backend's view of the cache: the Python `EngineCache` with its backend
/// reduced to what the cache uses of it. The key is the model's identity plus
/// the backend's tag, so two backends keep two artifacts per model and each
/// sees only its own.
public struct EngineCache: Sendable {
  public let layout: CacheLayout
  /// The backend's tag, for example `ort1.29.0.ane-Apple_A18_Pro`.
  public let tag: String
  /// The artifact's suffix: `.ortcache`, a directory, for onnxruntime.
  public let suffix: String
  /// The backend's name, recorded in last-loaded.json for the log.
  public let backend: String

  /// Makes engines/ and models/, as opening a Python EngineCache does.
  public init(layout: CacheLayout, tag: String, suffix: String, backend: String) {
    self.layout = layout
    self.tag = tag
    self.suffix = suffix
    self.backend = backend
    try? Files.makeDirectory(layout.engines)
    try? Files.makeDirectory(layout.models)
  }

  public func key(_ sha256: String) throws(RegistryError) -> String {
    try CacheLayout.validate(sha256: sha256)
    return "\(sha256.prefix(16)).\(tag)"
  }

  public func entry(_ sha256: String) throws(RegistryError) -> CacheEntry {
    let key = try key(sha256)
    return CacheEntry(path: layout.engines.appending(path: key + suffix), metaPath: layout.engines.appending(path: key + ".json"))
  }

  /// Model identities with an artifact this backend can load: a sidecar with
  /// a spec whose key is this backend's, and the artifact beside it.
  public func inventory() -> [String] {
    var found = Set<String>()
    for name in Files.names(in: layout.engines) where name.hasSuffix(".json") {
      let metaPath = layout.engines.appending(path: name)
      guard let sha256 = Files.readJSON(metaPath)?["spec"]?["sha256"]?.string, CacheLayout.isSHA256(sha256),
        let entry = try? entry(sha256), entry.metaPath == metaPath, entry.exists
      else { continue }
      found.insert(sha256)
    }
    return found.sorted()
  }

  /// Records what is loaded, for the next process to preload. frame_skip goes
  /// with it: the spec served is stamped with it, so preloading under another
  /// value hands the next client a spec it did not ask for.
  public func rememberLoaded(sha256: String, frameSkip: Int) {
    let marker: JSON = ["sha256": .string(sha256), "frame_skip": .int(Int64(frameSkip)), "backend": .string(backend)]
    // A cache that cannot be written still serves; it just cannot preload next time.
    try? Files.writeJSON(marker, to: layout.lastLoadedURL)
  }

  /// Keeps the newest few artifacts of this backend's kind; each is gigabytes.
  ///
  /// `protect` is never pruned whatever its mtime says: a Jetson boots at 1970
  /// without NTP, so a fresh build can look older than everything on disk.
  /// Other backends' artifacts are not touched.
  public func prune(keep: Int = CacheLayout.keepArtifacts, protect: URL? = nil) {
    let protected = protect.map(PythonPath.path)
    var found: [(url: URL, modified: Double)] = []
    for name in Files.names(in: layout.engines) where name.hasSuffix(suffix) {
      let url = layout.engines.appending(path: name)
      if let protected, PythonPath.path(url) == protected { continue }
      found.append((url, Files.status(url)?.modified ?? 0))
    }
    found.sort { $0.modified > $1.modified }
    let kept = max(keep - (protect == nil ? 0 : 1), 0)
    for (url, _) in found.dropFirst(kept) {
      CacheEntry(path: url, metaPath: url.deletingLastPathComponent().appending(path: PythonPath.withSuffix(url.lastPathComponent, ".json")))
        .remove()
    }
  }

  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    layout.sweepTemp(maxAge: maxAge)
  }
}

public struct CacheEntry: Sendable, Equatable {
  /// The artifact: a file for TensorRT, a directory for onnxruntime.
  public let path: URL
  public let metaPath: URL

  public init(path: URL, metaPath: URL) {
    self.path = path
    self.metaPath = metaPath
  }

  public var exists: Bool {
    Files.exists(path) && Files.isFile(metaPath)
  }

  /// Both halves, whichever exist. A directory artifact goes whole.
  public func remove() {
    Files.removeItem(path)
    try? Files.removeFile(metaPath)
  }
}

/// pathlib's name arithmetic, which the cache's keys are defined by.
enum PythonPath {
  /// `Path(name).suffix`: from the last dot, unless that dot starts or ends the name.
  static func suffix(_ name: String) -> String {
    guard let dot = name.lastIndex(of: "."), dot != name.startIndex, name.index(after: dot) != name.endIndex else { return "" }
    return String(name[dot...])
  }

  /// `Path(name).stem`.
  static func stem(_ name: String) -> String {
    let suffix = suffix(name)
    return String(name.dropLast(suffix.count))
  }

  /// `Path(name).with_suffix(new)`.
  static func withSuffix(_ name: String, _ new: String) -> String {
    stem(name) + new
  }

  /// The path pathlib compares: a directory's URL ends in a slash and the
  /// same name listed from its parent does not, but they are one path.
  static func path(_ url: URL) -> String {
    var path = url.standardizedFileURL.path(percentEncoded: false)
    while path.count > 1 && path.hasSuffix("/") { path.removeLast() }
    return path
  }
}
