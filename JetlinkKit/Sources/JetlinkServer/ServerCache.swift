import Foundation
import JetlinkRegistry

/// The server's cache: JetlinkRegistry's `EngineCache`, which the Python
/// conformance fixtures pin, for the backend this server runs. The backend
/// names its artifacts (tag, suffix, file or directory) once, here.
///
///     <root>/engines/<sha16>.<tag><suffix>    the artifact
///     <root>/engines/<sha16>.<tag>.json       its sidecar: build facts and the model spec
///     <root>/models/<sha16>.onnx              the model as uploaded or downloaded
public final class ServerCache: Sendable {
  public let layout: CacheLayout
  public let backend: any EngineBackend
  /// The artifacts kept (`Server.Configuration.keepPlans`).
  public let keep: Int
  private let store: EngineCache

  public init(root: URL, backend: any EngineBackend, keep: Int = CacheLayout.keepArtifacts) throws {
    layout = CacheLayout(root: root)
    self.backend = backend
    self.keep = keep
    for directory in [layout.engines, layout.models] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    store = EngineCache(layout: layout, tag: backend.tag(), suffix: backend.suffix, backend: backend.name)
  }

  /// The identity arrives from a peer and becomes a path, so anything but a
  /// digest is refused.
  public func entry(_ sha256: String) throws(RegistryError) -> CacheEntry {
    try store.entry(sha256)
  }

  public func modelPath(_ sha256: String) throws(RegistryError) -> URL {
    try layout.modelPath(sha256: sha256)
  }

  /// A request's identity was checked when it was made.
  func entry(_ request: Request) -> CacheEntry {
    try! store.entry(request.sha256)
  }

  func modelPath(_ request: Request) -> URL {
    try! layout.modelPath(sha256: request.sha256)
  }

  /// Model identities with artifacts this backend can load on this device.
  public func inventory() -> [String] {
    store.inventory()
  }

  /// What is loaded, for the next start to preload.
  public func rememberLoaded(_ sha256: String, frameSkip: Int) {
    store.rememberLoaded(sha256: sha256, frameSkip: frameSkip)
  }

  public func lastLoaded() -> LastLoaded? {
    layout.lastLoaded()
  }

  public func forgetLastLoaded() {
    try? FileManager.default.removeItem(at: layout.lastLoadedURL)
  }

  /// Removes the artifacts the backend can never load again, whatever their
  /// age: made by another runtime, or by an older preparation. They would
  /// only be rebuilt, and each holds gigabytes and a Neural Engine copy.
  public func pruneStale(protect: URL? = nil) {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: layout.engines.path(percentEncoded: false))) ?? []
    for name in names where name.hasSuffix(backend.suffix) {
      let url = layout.engines.appending(path: name)
      if let protect, url.standardizedFileURL.path == protect.standardizedFileURL.path { continue }
      if backend.isStale(artifactMeta: Artifact.sidecar(url)) {
        CacheEntry(path: url, metaPath: Artifact.sidecarURL(url)).remove()
      }
    }
  }

  /// Keep the newest few artifacts of this backend's kind. `protect` is never
  /// pruned whatever its mtime says.
  public func prune(protect: URL? = nil) {
    store.prune(keep: keep, protect: protect)
  }

  /// Build directories a killed build left behind.
  public func sweepTemp(maxAge: TimeInterval = 6 * 3600) {
    store.sweepTemp(maxAge: maxAge)
  }
}
