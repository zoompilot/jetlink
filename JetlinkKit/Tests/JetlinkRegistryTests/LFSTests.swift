import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// tests/test_registry.py's lfs section: the batch API, and comma's bucket,
/// which is read by oid. The endpoints' order is ConformanceTests'.
struct LFSTests {
  private let pointer = Pointer(oid: RegistryFixture.oid, size: RegistryFixture.size)
  private var gitlab: String { LFS.endpoints[1] }

  @Test func commasBucketIsAskedFirst() {
    #expect(LFS.endpoints.first == LFS.bucket)
  }

  @Test func theBatchAPIReturnsTheHref() async {
    let net = MockNet(["\(gitlab)/objects/batch": .body(RegistryFixture.data("lfs_batch_response.json"))])
    let href = await LFS.resolve(endpoint: gitlab, pointer: pointer, session: net.session)
    #expect(href?.hasPrefix("https://gitlab.com/commaai/openpilot-lfs.git/gitlab-lfs/objects/") == true)

    let call = net.calls.first
    #expect(call?.method == "POST")
    let body = call?.body.flatMap { try? JSON.parse($0) }
    #expect(body == ["operation": "download", "transfers": ["basic"], "objects": [["oid": .string(RegistryFixture.oid), "size": .int(RegistryFixture.size)]]])
  }

  @Test func theBatchAPIIsNilWhenTheServerLacksItOrIsDown() async {
    let missing = MockNet(["\(gitlab)/objects/batch": .body(RegistryFixture.data("lfs_batch_missing.json"))])
    #expect(await LFS.resolve(endpoint: gitlab, pointer: pointer, session: missing.session) == nil)
    #expect(await LFS.resolve(endpoint: gitlab, pointer: pointer, session: MockNet().session) == nil)
    let failing = MockNet(["\(gitlab)/objects/batch": .status(500)])
    #expect(await LFS.resolve(endpoint: gitlab, pointer: pointer, session: failing.session) == nil)
  }

  @Test func theBucketIsReadByOidAfterItsFirstByte() async {
    let url = "\(LFS.bucket)/resolve/sha256/\(RegistryFixture.oid)"
    #expect(LFS.bucketURL(oid: RegistryFixture.oid) == url)
    let net = MockNet([url: .body(Data("o".utf8))])
    #expect(await LFS.resolve(endpoint: LFS.bucket, pointer: pointer, session: net.session) == url)
    #expect(net.calls.map(\.method) == ["GET"])
    #expect(net.calls.first?.headers["Range"] == "bytes=0-0")

    let missing = MockNet([url: .status(404)])
    #expect(await LFS.resolve(endpoint: LFS.bucket, pointer: pointer, session: missing.session) == nil)
    #expect(await LFS.resolve(endpoint: LFS.bucket, pointer: pointer, session: MockNet().session) == nil)
  }

  @Test func anyBucketIsReadByOid() async {
    let other = "https://huggingface.co/buckets/someone/their-lfs"
    let url = "\(other)/resolve/sha256/\(RegistryFixture.oid)"
    let net = MockNet([url: .body(Data("o".utf8))])
    #expect(await LFS.resolve(endpoint: other, pointer: pointer, session: net.session) == url)
    #expect(!net.urls.contains { $0.hasSuffix("/objects/batch") })
  }

  @Test func aFetchTakesTheBucketsObjectWhenItHasIt() async throws {
    let tmp = try TemporaryDirectory()
    var routes = RegistryFixture.smallRoutes()
    routes[LFS.bucketURL(oid: RegistryFixture.blobSHA)] = .body(RegistryFixture.blob)
    let net = MockNet(routes)
    let path = try await Registry(layout: tmp.layout, session: net.session).fetch(RegistryFixture.smallRef)
    #expect(try Data(contentsOf: path) == RegistryFixture.blob)
    #expect(!net.urls.contains { $0.hasSuffix("/objects/batch") }, "no LFS server is asked past the bucket")
  }
}
