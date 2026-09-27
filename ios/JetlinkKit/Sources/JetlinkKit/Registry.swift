// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Which large models exist, which file each one is, and how to get the bytes:
// jetlink/registry, for a phone that fetches models over its own connection.
//
// sunnypilot's big-model catalog names each model by the comma openpilot
// commit it was built from. That commit's LFS pointer gives the ONNX's SHA-256
// and size, which is the identity the comma asks a server for; the bytes are
// on comma's LFS servers. State lives in <cache>/registry/ with the Python's
// file names and shapes, written atomically.

import CryptoKit
import Foundation

public enum RegistryConstants {
  public static let catalogURL = URL(
    string:
      "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v25.json"
  )!
  /// The selector version the fork requires (REQUIRED_JSON_VERSION on the comma).
  public static let requiredSelectorVersion = 19
  public static let defaultBigModelRef = "f877d7a0ccc3cce943c76e285214c020cd65c899"
  static let pointerURL =
    "https://raw.githubusercontent.com/commaai/openpilot/{ref}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
  static let lfsEndpoints = [
    "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs",
    "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs",
  ]
  static let lfsMediaType = "application/vnd.git-lfs+json"
  static let catalogMaxAge: TimeInterval = 3600
  static let pointerMax = 4096
  static let freeSlack = 64 << 20
}

public struct RegistryError: Error, CustomStringConvertible {
  public let description: String
  public init(_ d: String) { description = d }
}

public struct CatalogModel: Sendable, Identifiable, Equatable {
  public let name: String
  public let shortName: String
  public let ref: String
  public let buildTime: String
  public let index: Int
  public var sha256: String?
  public var bytes: Int?
  public var id: String { ref }
}

public struct Pointer: Sendable, Equatable {
  public let oid: String
  public let size: Int
}

public struct LocalModel: Sendable, Equatable {
  public let sha256: String
  public let bytes: Int
  public let name: String
  public let addedAt: Double
}

/// The big-model bundles, newest first. A malformed entry costs that entry only.
public func parseCatalog(_ data: JSON) -> [CatalogModel] {
  var found: [String: CatalogModel] = [:]
  for bundle in data["bundles"]?.array ?? [] {
    guard let ref = bundle["ref"]?.string, isRef(ref), found[ref] == nil else { continue }
    // minimum_selector_version is a string in the JSON; int() it as the Python does.
    let version: Int? = bundle["minimum_selector_version"]?.int ?? bundle["minimum_selector_version"]?.string.flatMap { Int($0) }
    guard version == RegistryConstants.requiredSelectorVersion, truthy(bundle["is_big"]) else { continue }
    let index = bundle["index"]?.int ?? bundle["index"]?.string.flatMap { Int($0) } ?? 0
    let display = bundle["display_name"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? String(ref.prefix(10))
    found[ref] = CatalogModel(
      name: display, shortName: bundle["short_name"]?.string ?? "", ref: ref, buildTime: bundle["build_time"]?.string ?? "",
      index: index, sha256: nil, bytes: nil)
  }
  return found.values.sorted { $0.index > $1.index }
}

private func truthy(_ v: JSON?) -> Bool {
  switch v {
  case .bool(let b)?: return b
  case .int(let i)?: return i != 0
  case .double(let d)?: return d != 0
  case .string(let s)?: return !s.isEmpty
  case .array(let a)?: return !a.isEmpty
  case .object(let o)?: return !o.isEmpty
  default: return false
  }
}

/// The oid and size in a git-lfs pointer's text, or nil if it is not one.
public func parsePointer(_ text: String) -> Pointer? {
  guard text.utf8.count <= RegistryConstants.pointerMax else { return nil }
  var oid: String?, size: Int?
  for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
    let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
    guard parts.count == 2 else { continue }
    let value = String(parts[1])
    if parts[0] == "oid" {
      oid = (value.hasPrefix("sha256:") ? String(value.dropFirst(7)) : value).trimmingCharacters(in: .whitespaces)
    } else if parts[0] == "size" {
      guard let n = Int(value.trimmingCharacters(in: .whitespaces)) else { return nil }
      size = n
    }
  }
  guard let oid, isSHA256(oid), let size, size > 0 else { return nil }
  return Pointer(oid: oid, size: size)
}

public final class Registry: @unchecked Sendable {
  public let root: URL
  let state: URL
  let models: URL
  let engines: URL
  private let lock = NSLock()
  private let log = JLogger("jetlink.registry")
  let session: URLSession

  public init(cacheRoot: URL, session: URLSession = .shared) throws {
    root = cacheRoot
    state = cacheRoot.appendingPathComponent("registry", isDirectory: true)
    models = cacheRoot.appendingPathComponent("models", isDirectory: true)
    engines = cacheRoot.appendingPathComponent("engines", isDirectory: true)
    self.session = session
    for d in [state, models] { try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true) }
  }

  var catalogPath: URL { state.appendingPathComponent("catalog.json") }
  var pointersPath: URL { state.appendingPathComponent("pointers.json") }
  var localPath: URL { state.appendingPathComponent("local-models.json") }

  // MARK: catalog

  public struct Catalog: Sendable {
    public var models: [CatalogModel]
    public var fetchedAt: Date?
    public var error: String?
  }

  /// The catalog, from the cache when it is under an hour old. A failed
  /// refresh returns the previous list with `error` set.
  public func catalog(refresh: Bool = false) async -> Catalog {
    let cached = readJSON(catalogPath)
    var raw = cached?["raw"]
    var fetchedAt = cached?["fetched_at"]?.double
    var failure: String?
    if refresh || raw == nil || fetchedAt == nil || Date().timeIntervalSince1970 - fetchedAt! > RegistryConstants.catalogMaxAge {
      do {
        let fresh = try await fetchJSON(RegistryConstants.catalogURL, timeout: 10)
        guard fresh.object != nil else { throw RegistryError("\(RegistryConstants.catalogURL) did not serve a JSON object") }
        raw = fresh
        fetchedAt = Date().timeIntervalSince1970
        writeJSON(
          catalogPath,
          ["fetched_at": .double(fetchedAt!), "url": .string(RegistryConstants.catalogURL.absoluteString), "raw": fresh])
      } catch {
        failure = "could not fetch the model catalog: \(error)"
      }
    }
    let pointers = self.pointers()
    let models = parseCatalog(raw ?? .object([:])).map { m -> CatalogModel in
      var m = m
      if let p = pointers[m.ref] {
        m.sha256 = p.oid
        m.bytes = p.size
      }
      return m
    }
    return Catalog(models: models, fetchedAt: fetchedAt.map { Date(timeIntervalSince1970: $0) }, error: failure)
  }

  /// Resolves the refs with no pointer yet, eight at a time. A failure costs its ref only.
  public func resolveMissing(_ refs: [String]) async {
    let known = pointers()
    let todo = Array(Set(refs.filter { isRef($0) && known[$0] == nil }))
    guard !todo.isEmpty else { return }
    var fresh: [String: Pointer] = [:]
    await withTaskGroup(of: (String, Pointer?).self) { group in
      var next = 0
      func add() {
        guard next < todo.count else { return }
        let ref = todo[next]
        next += 1
        group.addTask { [self] in (ref, try? await fetchPointer(ref)) }
      }
      for _ in 0..<min(8, todo.count) { add() }
      for await (ref, p) in group {
        if let p { fresh[ref] = p }
        add()
      }
    }
    if !fresh.isEmpty { savePointers(fresh) }
  }

  public func resolve(_ ref: String) async throws -> Pointer {
    guard isRef(ref) else { throw RegistryError("\(ref) is not a 40 character commit") }
    if let p = pointers()[ref] { return p }
    let p = try await fetchPointer(ref)
    savePointers([ref: p])
    log.info("\(ref.prefix(10)) is \(p.oid.prefix(16)), \(p.size >> 20) MB")
    return p
  }

  func fetchPointer(_ ref: String) async throws -> Pointer {
    let url = URL(string: RegistryConstants.pointerURL.replacingOccurrences(of: "{ref}", with: ref))!
    var request = URLRequest(url: url, timeoutInterval: 10)
    request.cachePolicy = .reloadIgnoringLocalCacheData
    let (data, response) = try await session.data(for: request)
    try check(response, url)
    guard data.count <= RegistryConstants.pointerMax, let p = parsePointer(String(decoding: data, as: UTF8.self)) else {
      throw RegistryError("\(ref.prefix(10)) did not serve an lfs pointer")
    }
    return p
  }

  // MARK: downloading

  /// Fetches one model's ONNX by catalog ref or SHA-256, from whichever LFS
  /// server has it, verifying size and hash before it takes its name.
  public func fetch(_ refOrSha: String, progress: @escaping @Sendable (Int, Int) -> Void) async throws -> URL {
    let pointer: Pointer
    if isRef(refOrSha) {
      pointer = try await resolve(refOrSha)
    } else if isSHA256(refOrSha), let p = pointers().values.first(where: { $0.oid == refOrSha }) {
      pointer = p
    } else {
      throw RegistryError("\(refOrSha.prefix(16)) is not a catalog ref or a known model")
    }
    let dest = try modelPath(pointer.oid)
    if fileSize(dest) == pointer.size { return dest }
    for endpoint in RegistryConstants.lfsEndpoints {
      guard let href = await lfsResolve(endpoint, pointer) else { continue }
      log.info("fetching \(pointer.oid.prefix(16)) (\(pointer.size >> 20) MB) from \(endpoint)")
      try await download(href, pointer, to: dest, progress: progress)
      return dest
    }
    throw RegistryError("no LFS server has \(pointer.oid.prefix(16))")
  }

  /// A download href from one LFS server, or nil if it lacks the object or is down.
  func lfsResolve(_ endpoint: String, _ pointer: Pointer) async -> URL? {
    var request = URLRequest(url: URL(string: "\(endpoint)/objects/batch")!, timeoutInterval: 30)
    request.httpMethod = "POST"
    request.setValue(RegistryConstants.lfsMediaType, forHTTPHeaderField: "Accept")
    request.setValue(RegistryConstants.lfsMediaType, forHTTPHeaderField: "Content-Type")
    let body: JSON = [
      "operation": "download", "transfers": ["basic"], "objects": [["oid": .string(pointer.oid), "size": .int(pointer.size)]],
    ]
    request.httpBody = body.data
    do {
      let (data, response) = try await session.data(for: request)
      try check(response, request.url!)
      for obj in try JSON.parse(data)["objects"]?.array ?? [] where obj["oid"]?.string == pointer.oid {
        if let e = obj["error"] {
          log.warning("\(endpoint) has no \(pointer.oid.prefix(16)) (\(e["message"]?.string ?? ""))")
          return nil
        }
        if let href = obj["actions"]?["download"]?["href"]?.string { return URL(string: href) }
      }
    } catch {
      log.warning("lfs batch failed at \(endpoint): \(error)")
    }
    return nil
  }

  /// Streams to a .part file, hashing as it goes, and only then renames it:
  /// a half-written model must never sit where a build could pick it up.
  func download(_ href: URL, _ pointer: Pointer, to dest: URL, progress: @escaping @Sendable (Int, Int) -> Void) async throws {
    let free = freeBytes(dest.deletingLastPathComponent())
    if free > 0 && free < pointer.size + RegistryConstants.freeSlack {
      throw RegistryError("need \(pointer.size >> 20) MB for the model, \(free >> 20) MB free")
    }
    let part = dest.appendingPathExtension("part")
    FileManager.default.createFile(atPath: part.path, contents: nil)
    guard let out = FileHandle(forWritingAtPath: part.path) else { throw RegistryError("cannot write \(part.lastPathComponent)") }
    let writer = StreamingDownload(out: out, total: pointer.size, progress: progress)
    do {
      try await writer.run(URLRequest(url: href, timeoutInterval: 30))
    } catch {
      try? out.close()
      try? FileManager.default.removeItem(at: part)
      throw error is CancellationError ? error : RegistryError("could not download \(pointer.oid.prefix(16)): \(error)")
    }
    try? out.close()
    let (written, digest) = writer.result
    guard written == pointer.size else {
      try? FileManager.default.removeItem(at: part)
      throw RegistryError("\(pointer.oid.prefix(16)) is \(written) bytes, expected \(pointer.size)")
    }
    guard digest == pointer.oid else {
      try? FileManager.default.removeItem(at: part)
      throw RegistryError("downloaded bytes hash to \(digest.prefix(16)), expected \(pointer.oid.prefix(16))")
    }
    _ = try? FileManager.default.removeItem(at: dest)
    try FileManager.default.moveItem(at: part, to: dest)
    progress(written, pointer.size)
  }

  // MARK: models on disk

  public func modelPath(_ sha256: String) throws -> URL {
    try EngineCache.validate(sha256)
    return models.appendingPathComponent("\(sha256.prefix(16)).onnx")
  }

  /// Takes a model file into the cache under its own identity. Progress is
  /// half hashing, half copying, as both read the whole file.
  public func importModel(_ source: URL, name: String?, progress: @escaping @Sendable (Double) -> Void) throws -> LocalModel {
    guard source.pathExtension.lowercased() == "onnx" else { throw RegistryError("\(source.lastPathComponent) is not an .onnx file") }
    let total = max(fileSize(source), 1)
    let (sha, n) = try sha256File(source.path, progress: { progress(0.5 * min(1, Double($0) / Double(total))) })
    let dest = try modelPath(sha)
    if fileSize(dest) != n {
      let part = dest.appendingPathExtension("part")
      try? FileManager.default.removeItem(at: part)
      do {
        try FileManager.default.copyItem(at: source, to: part)
        _ = try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: part, to: dest)
      } catch {
        try? FileManager.default.removeItem(at: part)
        throw RegistryError("could not copy \(source.lastPathComponent) into the cache: \(error)")
      }
    }
    progress(1.0)
    let local = LocalModel(sha256: sha, bytes: n, name: name ?? source.deletingPathExtension().lastPathComponent, addedAt: Date().timeIntervalSince1970)
    lock.lock()
    var records = localRecords().filter { $0["sha256"]?.string != sha }
    records.append(["sha256": .string(sha), "bytes": .int(n), "name": .string(local.name), "added_at": .double(local.addedAt)])
    writeJSON(localPath, .array(records))
    lock.unlock()
    return local
  }

  public func localModels() -> [LocalModel] {
    localRecords().compactMap { r in
      guard let sha = r["sha256"]?.string, let bytes = r["bytes"]?.int else { return nil }
      return LocalModel(sha256: sha, bytes: bytes, name: r["name"]?.string ?? "", addedAt: r["added_at"]?.double ?? 0)
    }
  }

  /// A human name for a model identity, from the catalog or an import.
  public func name(for sha256: String) -> String? {
    for (ref, p) in pointers() where p.oid == sha256 {
      if let raw = readJSON(catalogPath)?["raw"], let m = parseCatalog(raw).first(where: { $0.ref == ref }) { return m.name }
      return nil
    }
    return localModels().first { $0.sha256 == sha256 }?.name
  }

  /// Model files on disk: (full identity if known else the 16-character prefix, bytes).
  public func downloaded() -> [(sha256: String, bytes: Int)] {
    var known: [String: String] = [:]
    for p in pointers().values { known[String(p.oid.prefix(16))] = p.oid }
    for l in localModels() { known[String(l.sha256.prefix(16))] = l.sha256 }
    for u in (try? FileManager.default.contentsOfDirectory(at: engines, includingPropertiesForKeys: nil)) ?? []
    where u.pathExtension == "json" {
      if let sha = readJSON(u)?["spec"]?["sha256"]?.string, isSHA256(sha) { known[String(sha.prefix(16))] = sha }
    }
    return ((try? FileManager.default.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)) ?? [])
      .filter { $0.pathExtension == "onnx" && $0.deletingPathExtension().lastPathComponent.count == 16 }
      .map { u in
        let stem = u.deletingPathExtension().lastPathComponent
        return (known[stem] ?? stem, fileSize(u))
      }
      .sorted { $0.sha256 < $1.sha256 }
  }

  /// Deletes what was asked for; a file already gone is not an error.
  public func remove(_ sha256: String, artifacts: Bool, model: Bool) throws {
    try EngineCache.validate(sha256)
    let sha16 = String(sha256.prefix(16))
    let fm = FileManager.default
    if artifacts {
      for u in (try? fm.contentsOfDirectory(at: engines, includingPropertiesForKeys: nil)) ?? []
      where u.lastPathComponent.hasPrefix(sha16 + ".") {
        try? fm.removeItem(at: u)
      }
      if let last = readJSON(root.appendingPathComponent(EngineCache.lastLoadedName))?["sha256"]?.string, last == sha256 {
        try? fm.removeItem(at: root.appendingPathComponent(EngineCache.lastLoadedName))
      }
    }
    if model {
      let p = try modelPath(sha256)
      try? fm.removeItem(at: p)
      try? fm.removeItem(at: p.appendingPathExtension("part"))
      lock.lock()
      let records = localRecords()
      let kept = records.filter { $0["sha256"]?.string != sha256 }
      if kept.count != records.count { writeJSON(localPath, .array(kept)) }
      lock.unlock()
    }
  }

  // MARK: internals

  public func pointers() -> [String: Pointer] {
    var out: [String: Pointer] = [:]
    for (ref, v) in readJSON(pointersPath)?.object ?? [:] {
      if let oid = v["oid"]?.string, let size = v["size"]?.int, size > 0 { out[ref] = Pointer(oid: oid, size: size) }
    }
    return out
  }

  private func savePointers(_ fresh: [String: Pointer]) {
    lock.lock()
    defer { lock.unlock() }
    var known = readJSON(pointersPath)?.object ?? [:]
    for (ref, p) in fresh { known[ref] = ["oid": .string(p.oid), "size": .int(p.size)] }
    writeJSON(pointersPath, .object(known))
  }

  private func localRecords() -> [JSON] { (readJSON(localPath)?.array ?? []).filter { $0.object != nil } }

  private func readJSON(_ url: URL) -> JSON? { (try? Data(contentsOf: url)).flatMap { try? JSON.parse($0) } }

  private func writeJSON(_ url: URL, _ value: JSON) { try? atomicWrite(value.data, to: url) }

  private func fetchJSON(_ url: URL, timeout: TimeInterval) async throws -> JSON {
    var request = URLRequest(url: url, timeoutInterval: timeout)
    request.cachePolicy = .reloadIgnoringLocalCacheData
    let (data, response) = try await session.data(for: request)
    try check(response, url)
    return try JSON.parse(data)
  }

  private func check(_ response: URLResponse, _ url: URL) throws {
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      throw RegistryError("\(url.host ?? url.absoluteString) answered HTTP \(http.statusCode)")
    }
  }
}

public func freeBytes(_ url: URL) -> Int {
  #if os(iOS)
    if let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
      let n = v.volumeAvailableCapacityForImportantUsage
    {
      return Int(n)
    }
  #endif
  return (try? url.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity) ?? 0
}

/// One HTTP download written to a file handle and hashed as it arrives.
final class StreamingDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
  private let out: FileHandle
  private let total: Int
  private let progress: @Sendable (Int, Int) -> Void
  private var hasher = SHA256()
  private var written = 0
  private var lastReport = 0.0
  private var failure: Error?
  private var continuation: CheckedContinuation<Void, Error>?
  private let lock = NSLock()

  init(out: FileHandle, total: Int, progress: @escaping @Sendable (Int, Int) -> Void) {
    self.out = out
    self.total = total
    self.progress = progress
  }

  var result: (Int, String) {
    lock.lock()
    defer { lock.unlock() }
    return (written, hex(hasher.finalize()))
  }

  func run(_ request: URLRequest) async throws {
    let queue = OperationQueue()
    queue.maxConcurrentOperationCount = 1
    let session = URLSession(configuration: .default, delegate: self, delegateQueue: queue)
    defer { session.finishTasksAndInvalidate() }
    let task = session.dataTask(with: request)
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
        lock.lock()
        continuation = c
        lock.unlock()
        task.resume()
      }
    } onCancel: {
      task.cancel()
    }
  }

  func urlSession(
    _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
      lock.lock()
      failure = RegistryError("HTTP \(http.statusCode)")
      lock.unlock()
      completionHandler(.cancel)
      return
    }
    completionHandler(.allow)
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    lock.lock()
    defer { lock.unlock() }
    guard failure == nil else { return }
    do {
      try out.write(contentsOf: data)
      hasher.update(data: data)
      written += data.count
    } catch {
      failure = error
      dataTask.cancel()
      return
    }
    let now = monotonic()
    if now - lastReport > 0.25 {
      lastReport = now
      progress(written, total)
    }
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    lock.lock()
    let c = continuation
    continuation = nil
    let f = failure
    lock.unlock()
    if let f {
      c?.resume(throwing: f)
    } else if let error {
      c?.resume(throwing: (error as NSError).code == NSURLErrorCancelled ? CancellationError() : error)
    } else {
      c?.resume()
    }
  }
}
