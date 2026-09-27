// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Where built engines and uploaded models live, laid out as
// jetlink/server/cache.py lays them out:
//
//     <root>/engines/<sha16>.<backend tag>.ortcache/   the artifact
//     <root>/engines/<sha16>.<backend tag>.json        its sidecar: build facts and the model spec
//     <root>/models/<sha16>.onnx                       the model as uploaded or downloaded
//
// The key is the model's identity plus the backend's tag, so the Neural
// Engine and GPU builds of one model sit side by side.

import Foundation

public struct CacheEntry: Sendable {
  public let path: URL  // the artifact directory
  public let metaPath: URL

  public var exists: Bool {
    var isDir: ObjCBool = false
    return FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir) && isDir.boolValue
      && FileManager.default.fileExists(atPath: metaPath.path)
  }

  public func meta() throws -> JSON { try JSON.parse(Data(contentsOf: metaPath)) }

  public func writeMeta(_ meta: JSON) throws { try atomicWrite(meta.data, to: metaPath) }

  public func remove() {
    try? FileManager.default.removeItem(at: path)
    try? FileManager.default.removeItem(at: metaPath)
  }
}

public final class EngineCache: @unchecked Sendable {
  public let root: URL
  public let engines: URL
  public let models: URL
  /// Artifacts kept per compute unit. Each holds the prepared model and
  /// CoreML's conversion and compile, over 2 GB for a 766 MB model on a Mac,
  /// and a phone has less room than a Jetson.
  public var keep = 2

  static let lastLoadedName = "last-loaded.json"

  public init(root: URL) throws {
    self.root = root
    engines = root.appendingPathComponent("engines", isDirectory: true)
    models = root.appendingPathComponent("models", isDirectory: true)
    for d in [engines, models] {
      try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    }
  }

  public static func validate(_ sha256: String) throws {
    // Model identities arrive from the peer and become file names.
    if !isSHA256(sha256) { throw SpecError("model identity must be a lowercase SHA-256 digest") }
  }

  public func key(_ sha256: String, tag: String) throws -> String {
    try EngineCache.validate(sha256)
    return "\(sha256.prefix(16)).\(tag)"
  }

  public func entry(_ sha256: String, tag: String) throws -> CacheEntry {
    let k = try key(sha256, tag: tag)
    return CacheEntry(
      path: engines.appendingPathComponent(k + ".ortcache", isDirectory: true),
      metaPath: engines.appendingPathComponent(k + ".json"))
  }

  public func modelPath(_ sha256: String) throws -> URL {
    try EngineCache.validate(sha256)
    return models.appendingPathComponent("\(sha256.prefix(16)).onnx")
  }

  /// Model identities with an artifact for `tag`.
  public func inventory(tag: String) -> [String] {
    var found = Set<String>()
    for meta in (try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: nil)) ?? []
    where meta.pathExtension == "json" {
      guard let d = try? JSON.parse(Data(contentsOf: meta)), let sha = d["spec"]?["sha256"]?.string, isSHA256(sha),
        let e = try? entry(sha, tag: tag), e.metaPath.standardizedFileURL == meta.standardizedFileURL, e.exists
      else { continue }
      found.insert(sha)
    }
    return found.sorted()
  }

  /// What is loaded, for the next start to preload. frame_skip goes with it,
  /// as the spec served is stamped with it.
  public func rememberLoaded(_ sha256: String, frameSkip: Int, units: ComputeUnits) {
    let d: JSON = ["sha256": .string(sha256), "frame_skip": .int(frameSkip), "backend": "ort", "device": .string(units.rawValue)]
    try? atomicWrite(d.data, to: root.appendingPathComponent(EngineCache.lastLoadedName))
  }

  public func lastLoaded() -> (sha256: String, frameSkip: Int, units: ComputeUnits?)? {
    guard let d = try? JSON.parse(Data(contentsOf: root.appendingPathComponent(EngineCache.lastLoadedName))),
      let sha = d["sha256"]?.string, isSHA256(sha), let skip = d["frame_skip"]?.int
    else { return nil }
    return (sha, skip, d["device"]?.string.flatMap(ComputeUnits.init(rawValue:)))
  }

  public func forgetLastLoaded(_ sha256: String) {
    if lastLoaded()?.sha256 == sha256 {
      try? FileManager.default.removeItem(at: root.appendingPathComponent(EngineCache.lastLoadedName))
    }
  }

  /// Keeps the newest few artifacts with `tag`'s suffix; `protect` never goes.
  public func prune(tagSuffix: String, protect: URL?) {
    let fm = FileManager.default
    var found: [(URL, Date)] = []
    for u in (try? fm.contentsOfDirectory(at: engines, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
    where u.lastPathComponent.hasSuffix(tagSuffix + ".ortcache") {
      if let protect, u.standardizedFileURL == protect.standardizedFileURL { continue }
      let date = (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      found.append((u, date))
    }
    found.sort { $0.1 > $1.1 }
    for (u, _) in found.dropFirst(max(keep - (protect == nil ? 0 : 1), 0)) {
      CacheEntry(path: u, metaPath: u.deletingPathExtension().appendingPathExtension("json")).remove()
    }
  }

  /// Build directories a crashed or killed build left behind.
  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    let fm = FileManager.default
    for u in (try? fm.contentsOfDirectory(at: engines, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
    where u.lastPathComponent.hasPrefix("tmp") {
      let date = (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      if Date().timeIntervalSince(date) > maxAge { try? fm.removeItem(at: u) }
    }
  }

  /// Every sidecar in engines/, for listing what is on disk.
  public func sidecars() -> [(URL, JSON)] {
    ((try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: nil)) ?? [])
      .filter { $0.pathExtension == "json" }
      .compactMap { u in (try? JSON.parse(Data(contentsOf: u))).map { (u, $0) } }
  }
}

/// Writes through a temporary file and a rename, so a reader never sees half.
public func atomicWrite(_ data: Data, to url: URL) throws {
  try data.write(to: url, options: .atomic)
}

/// A version or device name as a filename component, as backends.base.sanitize.
public func sanitize(_ s: String) -> String {
  String(s.unicodeScalars.map { c in
    (("A"..."Z").contains(c) || ("a"..."z").contains(c) || ("0"..."9").contains(c) || c == "." || c == "_" || c == "-")
      ? Character(c) : "_"
  })
}

/// The bytes under a directory, as (outside `split`, inside it); missing is 0.
func treeBytes(_ root: URL, split: String? = nil) -> (Int, Int) {
  var outside = 0, inside = 0
  var stack: [(URL, Bool)] = [(root, false)]
  let fm = FileManager.default
  while let (dir, within) = stack.popLast() {
    guard let items = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey])
    else { continue }
    for u in items {
      let v = try? u.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
      if v?.isDirectory == true {
        stack.append((u, within || u.lastPathComponent == split))
      } else if within {
        inside += v?.fileSize ?? 0
      } else {
        outside += v?.fileSize ?? 0
      }
    }
  }
  return (outside, inside)
}
