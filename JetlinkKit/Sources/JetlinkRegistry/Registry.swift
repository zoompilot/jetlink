import Foundation
import JetlinkKit
import Synchronization

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif
#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif
#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// A model imported from disk rather than fetched by the model index.
public struct LocalModel: Sendable, Equatable {
  public let sha256: String
  public let bytes: Int64
  public let name: String
  /// Unix seconds, as the Python registry records it.
  public let addedAt: Double

  public init(sha256: String, bytes: Int64, name: String, addedAt: Double) {
    self.sha256 = sha256
    self.bytes = bytes
    self.name = name
    self.addedAt = addedAt
  }
}

/// Which large models exist, which one is which file, and how to get the
/// bytes. A port of `jetlink/registry`: the same model index, URLs, rules and
/// payloads, so the iPhone's server answers the control protocol as the
/// Python one did.
///
/// State lives in `<cache>/registry/`:
///
///     index.json         the last model index that parsed, as served; its mtime is when it was fetched
///     local-models.json  [{"sha256", "bytes", "name", "added_at"}] for models imported from disk
///
/// Each is written atomically, and every read-modify-write holds a lock,
/// because the control layer, a download and an index refresh can all be
/// writing while a comma is asking for a model. Calls are safe from any number
/// of tasks at once.
public final class Registry: Sendable {
  /// The suffix each backend gives its artifact, so a sidecar can be paired
  /// with the thing it describes without the backend (and so a Jetson's
  /// sidecars read on a phone).
  public static let artifactSuffixes = ["trt": ".plan", "tinygrad": ".pkl", "ort": ".ortcache", "litert": ".litertcache", "fake": ".fake"]

  static let hashChunk = 1 << 20
  static let copyChunk = 4 << 20

  public let layout: CacheLayout
  let session: URLSession
  let http: HTTP
  private let lock = Mutex(())
  /// The kept index as last parsed, and the file's (mtime, size) then: it is
  /// read for every name a payload carries, and parsed again only when it changed.
  private let kept = Mutex<(stamp: [Double], index: ModelIndex?)?>(nil)
  private static let log = Logger(subsystem: "io.zoompilot.jetlink", category: "registry")

  /// Opens the registry of a cache, making its registry and models directories.
  public init(layout: CacheLayout, session: URLSession = .shared) {
    self.layout = layout
    self.session = session
    self.http = HTTP(session: session)
    layout.makeRegistryDirectories()
  }

  // MARK: - the model index

  /// The kept index and when it was fetched, from disk alone; nil before the
  /// first fetch, or for a file that does not parse.
  public func modelIndex() -> ModelIndex? {
    keptIndex()?.index
  }

  private func keptIndex() -> (index: ModelIndex, fetchedAt: Double)? {
    guard let status = Files.status(layout.indexURL), status.isFile else { return nil }
    let stamp = [status.modified, Double(status.size)]
    if let memo = kept.withLock({ $0 }), memo.stamp == stamp {
      return memo.index.map { ($0, status.modified) }
    }
    let index = Files.readJSON(layout.indexURL).flatMap { try? ModelIndex.parse($0) }
    kept.withLock { $0 = (stamp, index) }
    return index.map { ($0, status.modified) }
  }

  /// The `catalog` event from disk alone, never the network: what a client
  /// gets the moment it connects. Nil when no index is kept; send
  /// `Registry.emptyCatalog` then and refresh off the connect path.
  public func cachedCatalog() -> CatalogEvent? {
    keptIndex().map { payload($0.index, fetchedAt: $0.fetchedAt, error: nil) }
  }

  /// The payload for a cache with no index in it yet.
  public static func emptyCatalog(error: String? = nil) -> CatalogEvent {
    CatalogEvent(fetchedAt: nil, url: ModelIndex.url, defaultRef: ModelIndex.defaultBigModelRef, error: error, models: [])
  }

  /// The `catalog` event: the model index, fetched again when the kept one
  /// is older than `maxAge`. A failed fetch, or an index that cannot be used,
  /// is not fatal: the previous list comes back with `error` set.
  public func catalog(refresh: Bool = false, maxAge: TimeInterval = ModelIndex.maxAge) async -> CatalogEvent {
    var current = keptIndex()
    var error: String?
    if refresh || current == nil || Date().timeIntervalSince1970 - (current?.fetchedAt ?? 0) > maxAge {
      do {
        current = try await fetchIndex()
      } catch let failure {
        error = failure.message
      }
    }
    guard let current else { return Registry.emptyCatalog(error: error) }
    return payload(current.index, fetchedAt: current.fetchedAt, error: error)
  }

  /// The index as served, kept only once it parses: `fetch_index` then the
  /// comma's `refresh_index`. A transport failure or a body that is not JSON
  /// is a network error, an index that cannot be used a registry error.
  private func fetchIndex() async throws(RegistryError) -> (index: ModelIndex, fetchedAt: Double) {
    let url = ModelIndex.url
    let body = try await http.get(url, timeout: ModelIndex.timeout)
    let data: JSON
    do {
      data = try JSON.parse(body)
    } catch {
      throw .network("\(url) did not serve JSON: \(error)")
    }
    let index = try ModelIndex.parse(data)
    var fetchedAt = Date().timeIntervalSince1970
    do {
      try body.write(to: layout.indexURL, options: .atomic)
      if let status = Files.status(layout.indexURL) {
        fetchedAt = status.modified
        kept.withLock { $0 = ([status.modified, Double(status.size)], index) }
      }
    } catch {
      Registry.log.warning("could not keep the model index: \(String(describing: error), privacy: .public)")
    }
    return (index, fetchedAt)
  }

  /// The models in the index's order. `index` counts down from the number
  /// listed, so a client that sorts by it, newest first, keeps that order.
  private func payload(_ index: ModelIndex, fetchedAt: Double?, error: String?) -> CatalogEvent {
    let count = index.models.count
    let models = index.models.enumerated().map { i, model in
      CatalogModel(
        name: model.name, shortName: model.shortName, ref: model.ref, buildTime: model.buildTime, index: count - i,
        sha256: model.oid, bytes: model.size, folder: model.folder)
    }
    // the index names its own default; the constant is for a cache with none
    let defaultRef = index.default ?? ModelIndex.defaultBigModelRef
    return CatalogEvent(fetchedAt: fetchedAt, url: ModelIndex.url, defaultRef: defaultRef, error: error, models: models)
  }

  // MARK: - lookups

  /// The ONNX a ref is, from the model index. A ref the kept index does not
  /// know fetches it again, once: the index may have learnt it since.
  public func resolve(ref: String) async throws(RegistryError) -> Pointer {
    guard CacheLayout.isRef(ref) else {
      throw .registry("\(JSON.pythonRepr(ref)) is not a 40 character commit")
    }
    return try await lookup({ $0.pointer(ref: ref) }, missing: "\(ref.prefix(10)) is not in the model index")
  }

  /// `find` in the kept index, else in one fetched again. A fetch that fails
  /// says so, as the kind of failure it was: offline is a network error.
  private func lookup<T>(_ find: (ModelIndex) -> T?, missing: String) async throws(RegistryError) -> T {
    if let index = modelIndex(), let found = find(index) { return found }
    let fresh: ModelIndex
    do {
      fresh = try await fetchIndex().index
    } catch {
      throw RegistryError(error.kind, "\(missing), and it could not be fetched again: \(error.message)")
    }
    guard let found = find(fresh) else { throw .registry(missing) }
    return found
  }

  /// The size the model index gives an ONNX, nil for one it does not know.
  /// Disk only.
  public func size(for sha256: String) -> Int64? {
    modelIndex()?.size(oid: sha256)
  }

  /// A human name and a ref for a model identity; either may be nil.
  public func name(for sha256: String) -> (name: String?, ref: String?) {
    nameLookup()(sha256)
  }

  /// `name(for:)` with the index and the local models read once, for a loop
  /// over many models. The name is the listed model's, else the import's;
  /// the ref is the listed model's, else any ref the index has for it.
  private func nameLookup() -> (String) -> (name: String?, ref: String?) {
    let index = modelIndex()
    let locals = localModels()
    return { sha256 in
      let name = index?.listed(oid: sha256)?.name ?? locals.first { $0.sha256 == sha256 }?.name
      return (name, index?.ref(oid: sha256))
    }
  }

  /// The ref whose model has this identity, from the model index. Disk only.
  public func ref(for sha256: String) -> String? {
    modelIndex()?.ref(oid: sha256)
  }

  // MARK: - models on disk

  /// Where the server looks for a model's ONNX.
  public func modelPath(sha256: String) throws(RegistryError) -> URL {
    try layout.modelPath(sha256: sha256)
  }

  /// Materialises one model's ONNX, from whichever LFS server has it.
  ///
  /// A ref or a sha256 is sized from the model index, because an LFS object
  /// is an oid plus a size. The servers are asked in order and the first with
  /// the object serves it. `progress` is called whenever the whole percent
  /// changes, and once with 1.0 at the end; `shouldStop` is asked between chunks.
  public func fetch(
    _ refOrSHA256: String,
    progress: @escaping @Sendable (Double) -> Void = { _ in },
    shouldStop: @escaping @Sendable () -> Bool = { false }
  ) async throws(RegistryError) -> URL {
    try await fetch(pointer: try await pointer(for: refOrSHA256), progress: progress, shouldStop: shouldStop)
  }

  /// `fetch(_:)` for a pointer known already: the comma names its model by
  /// the oid and size its own copy of the index gave it.
  public func fetch(
    pointer: Pointer,
    progress: @escaping @Sendable (Double) -> Void = { _ in },
    shouldStop: @escaping @Sendable () -> Bool = { false }
  ) async throws(RegistryError) -> URL {
    let dest = try layout.modelPath(sha256: pointer.oid)
    if let status = Files.status(dest), status.isFile, status.size == pointer.size {
      return dest
    }
    for endpoint in LFS.endpoints {
      guard let href = await LFS.resolve(endpoint: endpoint, pointer: pointer, session: session) else { continue }
      Registry.log.info("fetching \(pointer.oid.prefix(16), privacy: .public) (\(pointer.size >> 20) MB) from \(endpoint, privacy: .public)")
      return try await LFS.download(href: href, pointer: pointer, dest: dest, session: session, progress: progress, shouldStop: shouldStop)
    }
    throw .network("no LFS server has \(pointer.oid.prefix(16))")
  }

  /// The pointer for a ref or a sha256, from the model index. A sha256 the
  /// kept index does not size fetches it again, once, as a ref does.
  public func pointer(for refOrSHA256: String) async throws(RegistryError) -> Pointer {
    if CacheLayout.isRef(refOrSHA256) {
      return try await resolve(ref: refOrSHA256)
    }
    if CacheLayout.isSHA256(refOrSHA256) {
      let size = try await lookup(
        { $0.size(oid: refOrSHA256) }, missing: "size for \(refOrSHA256.prefix(16)) unknown: it is not in the model index")
      return Pointer(oid: refOrSHA256, size: size)
    }
    throw .registry("\(JSON.pythonRepr(refOrSHA256)) is neither a 40 character ref nor a 64 character sha256")
  }

  /// Takes a model from disk into the cache under its own identity.
  ///
  /// Progress covers the whole operation: the hashing pass is the first half
  /// and the copy the second, because both read the file end to end. A model
  /// already in the cache is not copied again, and a second import of it only
  /// renames it. The reading runs on a dispatch queue, off the task pool.
  public func importModel(
    at url: URL, name: String? = nil,
    progress: @escaping @Sendable (Double) -> Void = { _ in },
    shouldStop: @escaping @Sendable () -> Bool = { false }
  ) async throws(RegistryError) -> LocalModel {
    let fileName = url.lastPathComponent
    guard PythonPath.suffix(fileName).lowercased() == ".onnx" else {
      throw .registry("\(fileName) is not an .onnx file")
    }
    guard let status = Files.status(url), status.isFile else {
      throw .registry("\(url.path(percentEncoded: false)) does not exist")
    }
    let total = status.size
    let layout = self.layout
    let (sha256, nbytes) = try await Files.offload { () -> Result<(String, Int64), RegistryError> in
      do throws(RegistryError) {
        let (sha256, nbytes) = try Registry.hashFile(url, total: total, progress: progress, shouldStop: shouldStop)
        let dest = try layout.modelPath(sha256: sha256)
        if !(Files.status(dest).map { $0.isFile && $0.size == nbytes } ?? false) {
          try Registry.copyFile(url, to: dest, total: total, progress: progress, shouldStop: shouldStop)
        }
        return .success((sha256, nbytes))
      } catch {
        return .failure(error)
      }
    }.get()
    progress(1.0)

    let chosen = name.flatMap { $0.isEmpty ? nil : $0 } ?? PythonPath.stem(fileName)
    let local = LocalModel(sha256: sha256, bytes: nbytes, name: chosen, addedAt: Date().timeIntervalSince1970)
    try lock.withLock { _ throws(RegistryError) in
      var records = localRecords().filter { $0["sha256"] != .string(sha256) }
      records.append(["sha256": .string(local.sha256), "bytes": .int(local.bytes), "name": .string(local.name), "added_at": .double(local.addedAt)])
      do {
        try Files.writeJSON(.array(records.map(JSON.object)), to: layout.localModelsURL)
      } catch {
        throw .registry("could not record \(fileName): \(error.localizedDescription)")
      }
    }
    return local
  }

  public func localModels() -> [LocalModel] {
    localRecords().compactMap { record in
      guard let sha256 = record["sha256"]?.pythonString, let bytes = record["bytes"]?.pythonInt,
        let addedAt = Registry.pythonFloat(record["added_at"].flatMap { $0.truthy ? $0 : nil } ?? .double(0))
      else { return nil }
      let name = record["name"].flatMap { $0.truthy ? $0.pythonString : nil } ?? ""
      return LocalModel(sha256: sha256, bytes: bytes, name: name, addedAt: addedAt)
    }
  }

  private func localRecords() -> [JSONObject] {
    (Files.readJSON(layout.localModelsURL)?.array ?? []).compactMap(\.object)
  }

  /// `float(x)`, which also takes a numeric string.
  private static func pythonFloat(_ value: JSON) -> Double? {
    if let number = value.pythonNumber { return number }
    if let text = value.string { return Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) }
    return nil
  }

  /// The file's SHA-256 and size, read a megabyte at a time: an import's
  /// first pass, and the server's check of an upload.
  package static func hashFile(
    _ url: URL, total: Int64 = 0, progress: @Sendable (Double) -> Void = { _ in }, shouldStop: @Sendable () -> Bool = { false }
  ) throws(RegistryError) -> (String, Int64) {
    guard let handle = try? FileHandle(forReadingFrom: url) else {
      throw .registry("could not read \(url.path(percentEncoded: false))")
    }
    defer { try? handle.close() }
    var hasher = SHA256()
    var read: Int64 = 0
    while true {
      let chunk: Data
      do {
        chunk = try handle.read(upToCount: hashChunk) ?? Data()
      } catch {
        throw .registry("could not read \(url.path(percentEncoded: false)): \(error.localizedDescription)")
      }
      if chunk.isEmpty { break }
      if shouldStop() { throw .cancelled("import cancelled") }
      hasher.update(data: chunk)
      read += Int64(chunk.count)
      if total > 0 { progress(0.5 * min(1.0, Double(read) / Double(total))) }
    }
    return (hasher.finalize().map { String(format: "%02x", $0) }.joined(), read)
  }

  static func copyFile(
    _ source: URL, to dest: URL, total: Int64, progress: @Sendable (Double) -> Void, shouldStop: @Sendable () -> Bool
  ) throws(RegistryError) {
    let part = dest.deletingLastPathComponent().appending(path: dest.lastPathComponent + ".part")
    let fail = { (detail: String) -> RegistryError in
      .registry("could not copy \(source.path(percentEncoded: false)) into the cache: \(detail)")
    }
    do {
      guard FileManager.default.createFile(atPath: part.path(percentEncoded: false), contents: nil) else {
        throw fail("cannot create \(part.lastPathComponent)")
      }
      let input = try FileHandle(forReadingFrom: source)
      defer { try? input.close() }
      let output = try FileHandle(forWritingTo: part)
      defer { try? output.close() }
      var written: Int64 = 0
      while let chunk = try input.read(upToCount: copyChunk), !chunk.isEmpty {
        if shouldStop() { throw RegistryError.cancelled("import cancelled") }
        try output.write(contentsOf: chunk)
        written += Int64(chunk.count)
        if total > 0 { progress(0.5 + 0.5 * min(1.0, Double(written) / Double(total))) }
      }
    } catch {
      try? Files.removeFile(part)
      throw error as? RegistryError ?? fail(error.localizedDescription)
    }
    do {
      try Files.replace(part, with: dest)
    } catch {
      try? Files.removeFile(part)
      throw fail(error.localizedDescription)
    }
  }

  // MARK: - inventory and removal

  /// The `inventory` event: what is on disk and what it belongs to.
  ///
  /// A model file is named by the first 16 characters of its identity, so a
  /// file whose full identity is in no index, no local record and no sidecar
  /// is listed with a 16 character `sha256`: the prefix is all anyone knows.
  /// Artifacts of every backend are listed; `current` marks the one whose key
  /// is `<sha16>.<artifactTag>` with `artifactSuffix`, the one this server
  /// would load. With no tag nothing is current.
  public func inventory(artifactTag: String?, artifactSuffix: String, loaded: String?) -> InventoryEvent {
    let known = knownSHAs()
    let name = nameLookup()
    var models: [InventoryModel] = []
    var modelsBytes: Int64 = 0
    for fileName in Files.names(in: layout.models) where fileName.hasSuffix(".onnx") {
      let stem = PythonPath.stem(fileName)
      guard CacheLayout.isLowercaseHex(stem, count: 16) else { continue }
      let url = layout.models.appending(path: fileName)
      guard let status = Files.status(url) else { continue }
      let sha256 = known[stem] ?? stem
      let (modelName, ref) = sha256.utf8.count == 64 ? name(sha256) : (nil, nil)
      models.append(InventoryModel(sha256: sha256, bytes: status.size, path: url.path(percentEncoded: false), name: modelName, ref: ref))
      modelsBytes += status.size
    }

    var artifacts: [InventoryArtifact] = []
    var enginesBytes: Int64 = 0
    let engineNames = Files.names(in: layout.engines)
    for fileName in engineNames where fileName.hasSuffix(".json") {
      guard let entry = artifactEntry(sidecar: fileName, engineNames: engineNames, tag: artifactTag, suffix: artifactSuffix) else {
        continue
      }
      artifacts.append(entry)
      enginesBytes += entry.bytes
    }

    return InventoryEvent(
      loaded: loaded,
      lastLoaded: layout.lastLoaded()?.sha256,
      models: models,
      artifacts: artifacts,
      disk: InventoryDisk(modelsBytes: modelsBytes, enginesBytes: enginesBytes, freeBytes: Files.freeBytes(at: layout.root) ?? 0))
  }

  private func artifactEntry(sidecar: String, engineNames: [String], tag: String?, suffix: String) -> InventoryArtifact? {
    let metaPath = layout.engines.appending(path: sidecar)
    guard let meta = Files.readJSON(metaPath)?.object, let sha256 = meta["spec"]?["sha256"]?.string,
      CacheLayout.isSHA256(sha256)
    else { return nil }
    let key = PythonPath.stem(sidecar)
    let backend = meta["backend"].flatMap { $0.truthy ? $0.pythonString : nil } ?? ""
    guard let artifact = artifactPath(key: key, backend: backend, sidecar: sidecar, engineNames: engineNames) else { return nil }

    // Python's `or` chain: the first of these that is truthy, else trt_version
    // as it stands, which is how a Jetson's sidecar still reports its runtime.
    // LiteRT's is Swift's alone.
    let runtime = [meta["onnxruntime"], meta["tinygrad"], meta["litert"]].lazy.compactMap { $0 }.first { $0.truthy } ?? meta["trt_version"]
    let spec = meta["spec"].flatMap { $0.truthy ? $0 : nil }
    let current = tag.map { key == "\(sha256.prefix(16)).\($0)" && artifact.lastPathComponent.hasSuffix(suffix) } ?? false
    return InventoryArtifact(
      sha256: sha256,
      key: key,
      path: artifact.path(percentEncoded: false),
      bytes: Files.size(of: artifact),
      backend: backend,
      runtimeVersion: Registry.optionalString(runtime),
      device: meta["device"].flatMap { $0.truthy ? $0.pythonString : nil } ?? "",
      builtAt: Registry.optionalString(meta["built_at"]),
      buildSeconds: meta["build_seconds"]?.pythonNumber,
      checkpoint: Registry.optionalString(spec?["checkpoint"]),
      current: current)
  }

  /// The artifact a sidecar describes: the backend's own suffix first, then
  /// anything else under the same key.
  private func artifactPath(key: String, backend: String, sidecar: String, engineNames: [String]) -> URL? {
    if let suffix = Registry.artifactSuffixes[backend] {
      let candidate = layout.engines.appending(path: key + suffix)
      if Files.exists(candidate) { return candidate }
    }
    let name = engineNames.first { $0 != sidecar && $0.hasPrefix(key + ".") }
    return name.map { layout.engines.appending(path: $0) }
  }

  /// A JSON value where the event has an optional string: null and absent are nil.
  private static func optionalString(_ value: JSON?) -> String? {
    guard let value, value != .null else { return nil }
    return value.pythonString
  }

  /// prefix16 to full identity, from every place a full identity is written down.
  private func knownSHAs() -> [String: String] {
    var out: [String: String] = [:]
    for oid in modelIndex()?.sizes.keys.sorted() ?? [] {
      out[String(oid.prefix(16))] = oid
    }
    for local in localModels() where CacheLayout.isSHA256(local.sha256) {
      out[String(local.sha256.prefix(16))] = local.sha256
    }
    for fileName in Files.names(in: layout.engines) where fileName.hasSuffix(".json") {
      guard let sha256 = Files.readJSON(layout.engines.appending(path: fileName))?["spec"]?["sha256"]?.string,
        CacheLayout.isSHA256(sha256)
      else { continue }
      out[String(sha256.prefix(16))] = sha256
    }
    return out
  }

  /// Deletes what was asked for. A file that is already gone is not an error.
  ///
  /// The loaded engine is not this class's business: the control layer
  /// unloads first. Removing the artifacts also drops last-loaded.json when it
  /// names this model, and removing the model drops its import record, which
  /// would otherwise name a file that is gone.
  public func remove(sha256: String, artifacts: Bool, model: Bool) throws(RegistryError) {
    try CacheLayout.validate(sha256: sha256)
    let prefix = "\(sha256.prefix(16))."
    if artifacts {
      for fileName in Files.names(in: layout.engines) where fileName.hasPrefix(prefix) {
        let url = layout.engines.appending(path: fileName)
        if Files.isDirectory(url) {
          try? FileManager.default.removeItem(at: url)
        } else {
          do {
            try Files.removeFile(url)
          } catch {
            Registry.log.warning("could not remove \(url.path(percentEncoded: false), privacy: .public)")
          }
        }
      }
      if layout.lastLoaded()?.sha256 == sha256 {
        try? Files.removeFile(layout.lastLoadedURL)
      }
    }
    if model {
      let path = try layout.modelPath(sha256: sha256)
      do {
        try Files.removeFile(path)
        try Files.removeFile(path.deletingLastPathComponent().appending(path: path.lastPathComponent + ".part"))
      } catch {
        throw .registry("could not remove \(path.path(percentEncoded: false)): \(error.localizedDescription)")
      }
      try lock.withLock { _ throws(RegistryError) in
        let records = localRecords()
        let kept = records.filter { $0["sha256"] != .string(sha256) }
        guard kept.count != records.count else { return }
        do {
          try Files.writeJSON(.array(kept.map(JSON.object)), to: layout.localModelsURL)
        } catch {
          throw .registry("could not update the import records: \(error.localizedDescription)")
        }
      }
    }
  }
}
