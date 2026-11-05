import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// tests/fixtures/conformance/registry.json, which
/// JetlinkKit/Scripts/make_conformance_fixtures.py writes from the Python
/// registry: identities, model index parsing and the LFS endpoints. Its
/// `cache` block, the catalog and inventory payloads of one cache directory,
/// is Swift-owned and copied through.
extension JSON {
  fileprivate var int64: Int64? {
    if case .int(let value) = self { return value }
    return nil
  }

  fileprivate var bool: Bool? {
    if case .bool(let value) = self { return value }
    return nil
  }
}

struct RegistryConformanceTests {
  private var fixture: JSON { RegistryFixture.json("conformance/registry.json") }

  @Test func identitiesAreWhatPythonCallsThem() throws {
    for c in try #require(fixture["identities"]?.array) {
      let value = try #require(c["value"]?.string)
      #expect(CacheLayout.isRef(value) == c["is_ref"]?.bool, "\(value)")
      #expect(CacheLayout.isSHA256(value) == c["is_sha256"]?.bool, "\(value)")
    }
  }

  @Test func indexesParseAsPythonParsesThem() throws {
    let cases = try #require(fixture["parses"]?.array)
    #expect(cases.count > 20)
    for c in cases {
      let name = c["name"]?.string ?? "?"
      let data = c["index_file"]?.string.map { RegistryFixture.json($0) } ?? c["index"] ?? .null
      let parsed = try? ModelIndex.parse(data)
      guard let expected = c["expected"]?.object else {
        #expect(parsed == nil, "\(name): Python refuses it whole")
        continue
      }
      let index = try #require(parsed, "\(name): Python parses it")
      let models = (expected["models"]?.array ?? []).map { m in
        ModelIndex.Model(
          ref: m["ref"]?.string ?? "", oid: m["oid"]?.string ?? "", size: m["size"]?.int64 ?? -1, name: m["name"]?.string ?? "",
          shortName: m["short_name"]?.string ?? "", folder: m["folder"]?.string ?? "", buildTime: m["build_time"]?.string ?? "",
          pr: m["pr"]?.int64, is20Hz: m["is_20hz"]?.bool ?? false, generation: m["generation"]?.string ?? "",
          environment: m["environment"]?.string ?? "", runner: m["runner"]?.string ?? "")
      }
      #expect(index.default == expected["default"]?.string, "\(name)")
      #expect(index.models == models, "\(name)")
      var refs: [String: Pointer] = [:]
      for (ref, p) in expected["refs"]?.object ?? JSONObject() {
        refs[ref] = Pointer(oid: p["oid"]?.string ?? "", size: p["size"]?.int64 ?? -1)
      }
      #expect(index.refs == refs, "\(name)")
      var sizes: [String: Int64] = [:]
      for (oid, size) in expected["sizes"]?.object ?? JSONObject() {
        sizes[oid] = size.int64 ?? -1
      }
      #expect(index.sizes == sizes, "\(name)")
    }
  }

  @Test func theEndpointsAreAskedInPythonsOrder() throws {
    let endpoints = try #require(fixture["endpoints"]?.array).compactMap(\.string)
    #expect(endpoints.count == 4)
    #expect(LFS.endpoints == endpoints)
  }

  /// The cache directory the Python registry was shown, written again here.
  private func writeTree(_ tree: JSONObject, under root: URL) throws {
    for (relative, content) in tree {
      let url = root.appending(path: relative)
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      if let json = content["json"] {
        try json.data().write(to: url)
      } else if let text = content["text"]?.string {
        try Data(text.utf8).write(to: url)
      } else {
        try Data(count: Int(content["bytes"]?.int64 ?? 0)).write(to: url)
      }
      // the kept index's mtime is when it was fetched
      if let mtime = content["mtime"]?.pythonNumber {
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: mtime)], ofItemAtPath: url.path(percentEncoded: false))
      }
    }
  }

  /// The payload as the Python fixture spells it: the root as $ROOT, no free space.
  private func normalized<T: Encodable>(_ value: T, root: URL) throws -> Any {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.withoutEscapingSlashes]
    var text = String(decoding: try encoder.encode(value), as: UTF8.self)
    for prefix in [root.path, root.resolvingSymlinksInPath().path] {
      text = text.replacingOccurrences(of: prefix, with: "$ROOT")
    }
    var object = try JSONSerialization.jsonObject(with: Data(text.utf8))
    if var dict = object as? [String: Any], var disk = dict["disk"] as? [String: Any] {
      disk["free_bytes"] = 0
      dict["disk"] = disk
      object = dict
    }
    return object
  }

  @Test func oneCacheDirectoryReadsTheSameFromBothSides() throws {
    let raw = try Data(contentsOf: RegistryFixture.directory.appending(path: "conformance/registry.json"))
    let python = try #require((try JSONSerialization.jsonObject(with: raw) as? [String: Any])?["cache"] as? [String: Any])
    let cache = try #require(fixture["cache"]?.object)
    let tmp = try TemporaryDirectory()
    try writeTree(try #require(cache["tree"]?.object), under: tmp.url)
    let registry = Registry(layout: tmp.layout)

    let catalog = try #require(registry.cachedCatalog())
    var comparison = JSONComparison()
    comparison.compare(python: python["catalog"]!, swift: try normalized(catalog, root: tmp.url), at: "catalog")
    let inventory = registry.inventory(
      artifactTag: cache["artifact_tag"]?.string, artifactSuffix: cache["artifact_suffix"]?.string ?? "", loaded: nil)
    comparison.compare(python: python["inventory"]!, swift: try normalized(inventory, root: tmp.url), at: "inventory")
    #expect(comparison.differences.isEmpty, "\(comparison.differences)")
  }
}
