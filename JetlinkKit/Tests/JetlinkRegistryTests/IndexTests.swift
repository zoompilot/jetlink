import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// tests/test_registry.py's index section, and what the server keeps of it:
/// the last index that parsed, the catalog payload made of it, and every
/// lookup it answers. The parse itself is held to the Python by
/// ConformanceTests.
struct IndexParseTests {
  private var published: JSON { RegistryFixture.json(RegistryFixture.indexFile) }

  @Test func thePublishedIndexParses() throws {
    let index = try ModelIndex.parse(published)
    #expect(index.models.count == RegistryFixture.listed)
    #expect(index.models.first?.ref == RegistryFixture.newestRef)
    #expect(Set(index.models.map(\.folder)) == ["2026 Deep RL Models", "Master Models"])
    #expect(index.pointer(ref: RegistryFixture.ref) == Pointer(oid: RegistryFixture.oid, size: RegistryFixture.size))
    #expect(index.pointer(ref: ModelIndex.defaultBigModelRef) != nil, "the default model is one the index resolves")
  }

  @Test func aRefIsListedUnderItsONNX() throws {
    let index = try ModelIndex.parse(published)
    #expect(index.pointer(ref: RegistryFixture.unlistedRef)?.oid == RegistryFixture.oid)
    #expect(index.listed(ref: RegistryFixture.unlistedRef)?.ref == RegistryFixture.ref)
    #expect(index.listed(ref: RegistryFixture.ref)?.name == RegistryFixture.name)
    #expect(index.listed(ref: String(repeating: "f", count: 40)) == nil)
  }

  @Test func anUnlistedONNXIsNamedByTheFirstRefTheIndexHasForIt() throws {
    let index = try ModelIndex.parse([
      "schema": 1,
      "refs": [
        String(repeating: "c", count: 40): ["oid": .string(String(repeating: "e", count: 64)), "size": 5],
        String(repeating: "b", count: 40): ["oid": .string(String(repeating: "e", count: 64)), "size": 5],
      ],
      "models": [["ref": .string(RegistryFixture.ref), "oid": .string(RegistryFixture.oid), "size": 9]],
    ])
    #expect(index.ref(oid: String(repeating: "e", count: 64)) == String(repeating: "c", count: 40))
    #expect(index.ref(oid: RegistryFixture.oid) == RegistryFixture.ref)
    #expect(index.ref(oid: String(repeating: "d", count: 64)) == nil)
    #expect(index.size(oid: String(repeating: "e", count: 64)) == 5)
  }

  @Test func anIndexItCannotUseSaysWhy() {
    let cases: [(JSON, String)] = [
      (["models"], "the index is not a JSON object"),
      (["schema": 2, "models": []], "the index is schema 2, not 1"),
      (["schema": "1"], "the index is schema '1', not 1"),
      (["models": []], "the index is schema None, not 1"),
      (["schema": 1, "models": [1, "two"]], "the index lists no model"),
    ]
    for (data, why) in cases {
      #expect(throws: RegistryError.registry(why)) { try ModelIndex.parse(data) }
    }
  }

  @Test func aSchemaPythonCallsOneIsOne() throws {
    let model: JSON = ["ref": .string(RegistryFixture.ref), "oid": .string(RegistryFixture.oid), "size": 9]
    for schema: JSON in [1, .double(1), .bool(true)] {
      #expect(try ModelIndex.parse(["schema": schema, "models": [model]]).models.count == 1)
    }
  }
}

struct IndexPayloadTests {
  @Test func isCachedUntilItGoesStale() async throws {
    let net = MockNet(RegistryFixture.indexRoutes())
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: net.session)

    let payload = await registry.catalog()
    #expect(payload.models.count == RegistryFixture.listed)
    #expect(payload.error == nil)
    #expect(payload.fetchedAt != nil)
    #expect(payload.defaultRef == "bf3e3631b3f91d92a1020a5e0dd4298b93ff4244")
    #expect(payload.url == ModelIndex.url)

    _ = await registry.catalog()
    #expect(net.count(ModelIndex.url) == 1, "a fresh index must not go to the network")

    _ = await registry.catalog(maxAge: -1)
    #expect(net.count(ModelIndex.url) == 2)

    _ = await registry.catalog(refresh: true)
    #expect(net.count(ModelIndex.url) == 3)
  }

  @Test func everyModelCarriesItsONNXAndFolderInTheIndexsOrder() async throws {
    let tmp = try TemporaryDirectory()
    let payload = await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session).catalog()
    #expect(payload.models.allSatisfy { $0.sha256 != nil && $0.bytes != nil && $0.folder != nil })
    #expect(payload.models.map(\.index) == Array((1...RegistryFixture.listed).reversed()), "newest first, by index")
    let entry = try #require(payload.models.first { $0.ref == RegistryFixture.ref })
    #expect(entry.sha256 == RegistryFixture.oid && entry.bytes == RegistryFixture.size)
    #expect(entry.name == RegistryFixture.name && entry.shortName == "BMV4" && entry.folder == "Master Models")
  }

  @Test func theIndexIsKeptAsServed() async throws {
    let tmp = try TemporaryDirectory()
    _ = await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session).catalog()
    #expect(tmp.names("registry") == ["index.json"])
    #expect(try Data(contentsOf: tmp.layout.indexURL) == RegistryFixture.data(RegistryFixture.indexFile))
    let fetchedAt = try #require(Registry(layout: tmp.layout).cachedCatalog()?.fetchedAt)
    #expect(abs(Date().timeIntervalSince1970 - fetchedAt) < 60, "the file's mtime is when it was fetched")
  }

  @Test func aFailedRefreshKeepsThePreviousIndex() async throws {
    let tmp = try TemporaryDirectory()
    _ = await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session).catalog()

    let payload = await Registry(layout: tmp.layout, session: MockNet().session).catalog(refresh: true)

    #expect(payload.error?.contains(ModelIndex.url) == true)
    #expect(payload.models.count == RegistryFixture.listed)
    #expect(payload.fetchedAt != nil)
  }

  @Test(arguments: [
    (#"{"schema": 2, "models": []}"#, "the index is schema 2, not 1"),
    (#"{"schema": 1, "models": []}"#, "the index lists no model"),
    ("[1, 2]", "the index is not a JSON object"),
  ])
  func anIndexItCannotUseKeepsThePreviousOne(body: String, why: String) async throws {
    let tmp = try TemporaryDirectory()
    _ = await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session).catalog()

    let net = MockNet([ModelIndex.url: .body(Data(body.utf8))])
    let payload = await Registry(layout: tmp.layout, session: net.session).catalog(refresh: true)

    #expect(payload.error == why)
    #expect(payload.models.count == RegistryFixture.listed)
    #expect(try Data(contentsOf: tmp.layout.indexURL) == RegistryFixture.data(RegistryFixture.indexFile), "the kept copy is untouched")
  }

  @Test func aBodyThatIsNotJSONIsANetworkError() async throws {
    let tmp = try TemporaryDirectory()
    let broken = MockNet([ModelIndex.url: .body(Data("<html>".utf8))])
    let payload = await Registry(layout: tmp.layout, session: broken.session).catalog()
    #expect(payload.error?.hasPrefix("\(ModelIndex.url) did not serve JSON") == true)
    #expect(!Files.exists(tmp.layout.indexURL))
  }

  @Test func anEmptyCacheAndNoNetworkIsAnEmptyList() async throws {
    let tmp = try TemporaryDirectory()
    let payload = await Registry(layout: tmp.layout, session: MockNet().session).catalog()
    #expect(payload.models.isEmpty)
    #expect(payload.fetchedAt == nil)
    #expect(payload.error?.isEmpty == false)
  }

  @Test func theCachedCatalogNeverTouchesTheNetwork() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(RegistryFixture.indexRoutes())
    let registry = Registry(layout: tmp.layout, session: net.session)
    #expect(registry.cachedCatalog() == nil)
    #expect(net.calls.isEmpty)

    let fetched = await registry.catalog()
    let calls = net.calls.count
    #expect(registry.cachedCatalog() == fetched)
    #expect(Registry(layout: tmp.layout).cachedCatalog() == fetched, "the same from disk alone")
    #expect(net.calls.count == calls)

    let empty = Registry.emptyCatalog(error: "offline")
    #expect(empty.models.isEmpty && empty.fetchedAt == nil && empty.error == "offline")
    #expect(empty.url == ModelIndex.url && empty.defaultRef == ModelIndex.defaultBigModelRef)
  }

  @Test func aKeptIndexThatDoesNotParseIsNoIndex() throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout)
    try Data("not json".utf8).write(to: tmp.layout.indexURL)
    #expect(registry.cachedCatalog() == nil && registry.modelIndex() == nil)
    try RegistryFixture.data(RegistryFixture.indexFile).write(to: tmp.layout.indexURL)
    #expect(registry.cachedCatalog()?.models.count == RegistryFixture.listed, "a file that changed is read again")
  }

  @Test(arguments: [
    (#""f877d7a0ccc3cce943c76e285214c020cd65c899""#, "f877d7a0ccc3cce943c76e285214c020cd65c899"),
    (#""4adbb85742b465108ac579da680bcc633eedb2cc""#, ModelIndex.defaultBigModelRef),  // resolved, not listed
    ("7", ModelIndex.defaultBigModelRef),
    ("null", ModelIndex.defaultBigModelRef),
  ])
  func theDefaultIsTheIndexsOwnWhenItListsIt(named: String, expected: String) async throws {
    let tmp = try TemporaryDirectory()
    let refs = #""4adbb85742b465108ac579da680bcc633eedb2cc": {"oid": "\#(RegistryFixture.oid)", "size": 9}"#
    let model = #"{"ref": "\#(RegistryFixture.ref)", "oid": "\#(RegistryFixture.oid)", "size": 9}"#
    let body = #"{"schema": 1, "default": \#(named), "refs": {\#(refs)}, "models": [\#(model)]}"#
    let payload = await Registry(layout: tmp.layout, session: MockNet([ModelIndex.url: .body(Data(body.utf8))]).session).catalog()
    #expect(payload.error == nil)
    #expect(payload.defaultRef == expected)
  }

  @Test func theEventKeepsItsShapeAndAddsTheFolder() async throws {
    let tmp = try TemporaryDirectory()
    let payload = await Registry(layout: tmp.layout, session: MockNet(RegistryFixture.indexRoutes()).session).catalog()
    let object = ControlEvent.catalog(payload).payload()
    #expect(Set(object.keys) == ["fetched_at", "url", "default_ref", "models"])
    let row = try #require((object["models"] as? [[String: Any]])?.first)
    #expect(Set(row.keys) == ["name", "short_name", "ref", "build_time", "index", "sha256", "bytes", "folder"])
    // a client from before the folder still reads the event, and one with it reads an old server's
    let old = Data(#"{"name": "n", "short_name": "", "ref": "r", "build_time": "", "index": 1}"#.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    #expect(try decoder.decode(CatalogModel.self, from: old).folder == nil)
  }
}

struct IndexLookupTests {
  private func registry(_ routes: [String: MockNet.Reply] = RegistryFixture.indexRoutes()) async throws -> (Registry, MockNet, TemporaryDirectory) {
    let tmp = try TemporaryDirectory()
    let net = MockNet(routes)
    let registry = Registry(layout: tmp.layout, session: net.session)
    _ = await registry.catalog()
    return (registry, net, tmp)
  }

  @Test func resolveIsALookup() async throws {
    let (registry, net, tmp) = try await registry()
    defer { withExtendedLifetime(tmp) {} }
    let calls = net.calls.count
    #expect(try await registry.resolve(ref: RegistryFixture.ref) == Pointer(oid: RegistryFixture.oid, size: RegistryFixture.size))
    #expect(try await registry.resolve(ref: RegistryFixture.unlistedRef) == Pointer(oid: RegistryFixture.oid, size: RegistryFixture.size))
    #expect(net.calls.count == calls, "a ref the index knows touches no network")
  }

  @Test func aRefTheKeptIndexLacksFetchesItOnceAgain() async throws {
    let (registry, net, tmp) = try await registry()
    defer { withExtendedLifetime(tmp) {} }
    let unknown = String(repeating: "f", count: 40)
    let error = await #expect(throws: RegistryError.self) { try await registry.resolve(ref: unknown) }
    #expect(error?.message == "ffffffffff is not in the model index")
    #expect(net.count(ModelIndex.url) == 2)
  }

  @Test func resolveRefusesSomethingThatIsNotACommit() async throws {
    let (registry, net, tmp) = try await registry()
    defer { withExtendedLifetime(tmp) {} }
    let error = await #expect(throws: RegistryError.self) { try await registry.resolve(ref: "nothex") }
    #expect(error?.message == "'nothex' is not a 40 character commit")
    #expect(net.count(ModelIndex.url) == 1)
  }

  @Test func namesAndRefsComeFromTheIndex() async throws {
    let (registry, _, tmp) = try await registry()
    defer { withExtendedLifetime(tmp) {} }
    #expect(registry.name(for: RegistryFixture.oid) == (RegistryFixture.name, RegistryFixture.ref))
    #expect(registry.ref(for: RegistryFixture.oid) == RegistryFixture.ref)
    #expect(registry.size(for: RegistryFixture.oid) == RegistryFixture.size)
    let other = String(repeating: "b", count: 64)
    #expect(registry.name(for: other) == (nil, nil))
    #expect(registry.ref(for: other) == nil && registry.size(for: other) == nil)
    // an ONNX comma published that no ref names is sized all the same
    let loose = "01effc679eb46d6036ce0302673a87fe2dcfcc95a679daeeb929679e53c3647a"
    #expect(registry.size(for: loose) != nil && registry.ref(for: loose) == nil)
  }

  @Test func anImportTheIndexDoesNotNameKeepsItsOwnName() async throws {
    let (registry, _, tmp) = try await registry()
    let file = tmp.url.appending(path: "mine.onnx")
    try Data("mine".utf8).write(to: file)
    let local = try await registry.importModel(at: file)
    #expect(registry.name(for: local.sha256) == ("mine", nil))
  }

  @Test func aKnownSHA256IsFetchedBySizeFromTheIndex() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: MockNet(RegistryFixture.smallRoutes()).session)
    let path = try await registry.fetch(RegistryFixture.blobSHA)
    #expect(try Data(contentsOf: path) == RegistryFixture.blob)
  }

  @Test func offlineAndUnknownIsANetworkError() async throws {
    let tmp = try TemporaryDirectory()
    let error = await #expect(throws: RegistryError.self) {
      try await Registry(layout: tmp.layout, session: MockNet().session).resolve(ref: RegistryFixture.ref)
    }
    #expect(error?.isNetwork == true)
    #expect(error?.message.hasPrefix("f877d7a0cc is not in the model index, and it could not be fetched again: could not fetch") == true)
  }

  @Test func aSHA256TheIndexDoesNotKnowIsRefused() async throws {
    let (registry, _, tmp) = try await registry()
    defer { withExtendedLifetime(tmp) {} }
    let unknown = await #expect(throws: RegistryError.self) { try await registry.fetch(String(repeating: "b", count: 64)) }
    #expect(unknown?.message == "size for bbbbbbbbbbbbbbbb unknown: it is not in the model index")
    let neither = await #expect(throws: RegistryError.self) { try await registry.fetch("nothex") }
    #expect(neither?.message == "'nothex' is neither a 40 character ref nor a 64 character sha256")
  }
}
