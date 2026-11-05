import Foundation
import JetlinkRegistry

/// The registry's fixtures, the Python suite's own: <repo>/tests/fixtures,
/// and the routes a MockNet answers them on.
public enum RegistryFixture {
  public static let directory = SourceTree.root().appending(path: "tests/fixtures")
  /// The model index as zoompilot/jetlink-model-index published it.
  public static let indexFile = "model_index_v1.json"
  /// How many models it lists.
  public static let listed = 19
  /// One of them: its ref, and its ONNX.
  public static let ref = "f877d7a0ccc3cce943c76e285214c020cd65c899"
  public static let oid = "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525"
  public static let size: Int64 = 765_953_504
  public static let name = "BMRLNAP Model v4 (August 30, 2026)"
  /// A recompile of `ref`'s ONNX, which the index resolves and does not list.
  public static let unlistedRef = "4adbb85742b465108ac579da680bcc633eedb2cc"
  /// The model it lists first.
  public static let newestRef = "b92fc4828664f8767bce5e92f1630ed319cbe509"
  /// A ref whose ONNX is `blob`.
  public static let smallRef = String(repeating: "a", count: 40)
  public static let blob = Data(String(repeating: "onnx", count: 1024).utf8)
  public static let blobSHA = "9cf0cb741f1509847a90129514b5de3f9fff85e2e40de72369f516e9d7ed5b04"

  /// A fixture's bytes; none when it is missing, which the test then fails on.
  public static func data(_ name: String) -> Data {
    (try? Data(contentsOf: directory.appending(path: name))) ?? Data()
  }

  /// The published index at `ModelIndex.url`, and `extra`.
  public static func indexRoutes(_ extra: [String: MockNet.Reply] = [:]) -> [String: MockNet.Reply] {
    var routes: [String: MockNet.Reply] = [ModelIndex.url: .body(data(indexFile))]
    routes.merge(extra) { _, new in new }
    return routes
  }

  /// An index listing `smallRef` alone, as `oid` and `size`.
  public static func smallIndex(oid: String = blobSHA, size: Int64 = Int64(blob.count)) -> Data {
    let model: [String: Any] = ["ref": smallRef, "oid": oid, "size": size, "name": "Small (November 05, 2026)", "folder": "Test Models"]
    return (try? JSONSerialization.data(withJSONObject: ["schema": 1, "models": [model]] as [String: Any])) ?? Data()
  }

  /// A batch response fixture, retargeted at `oid` and served from `href`;
  /// the one that has no object without an `href`.
  public static func batch(oid: String, size: Int64, href: String?) -> Data {
    guard
      var payload = (try? JSONSerialization.jsonObject(with: data(href == nil ? "lfs_batch_missing.json" : "lfs_batch_response.json"))) as? [String: Any],
      var objects = payload["objects"] as? [[String: Any]], !objects.isEmpty
    else { return Data() }
    objects[0]["oid"] = oid
    objects[0]["size"] = size
    if let href {
      objects[0]["actions"] = ["download": ["href": href, "header": [String: String]()]]
    }
    payload["objects"] = objects
    return (try? JSONSerialization.data(withJSONObject: payload)) ?? Data()
  }

  /// The index lists `smallRef` as `body` under `oid`; comma's bucket and
  /// the first LFS server have nothing, the next serves it from `href`, which
  /// a local server may answer itself.
  public static func smallRoutes(
    _ body: Data = blob, oid: String = blobSHA, size: Int64? = nil, href: String = "https://blob.example/object"
  ) -> [String: MockNet.Reply] {
    let size = size ?? Int64(body.count)
    var routes: [String: MockNet.Reply] = [
      ModelIndex.url: .body(smallIndex(oid: oid, size: size)),
      LFS.bucketURL(oid: oid): .status(404),
      "\(LFS.endpoints[1])/objects/batch": .body(batch(oid: oid, size: size, href: nil)),
      "\(LFS.endpoints[2])/objects/batch": .body(batch(oid: oid, size: size, href: href)),
    ]
    if !href.hasPrefix("http://127.0.0.1") {
      routes[href] = .body(body)
    }
    return routes
  }
}
