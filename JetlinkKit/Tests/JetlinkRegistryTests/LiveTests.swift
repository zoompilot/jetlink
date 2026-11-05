import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// Against the real servers, and only when asked: JETLINK_LIVE=1 swift test
/// --filter LiveTests. It fetches the model index into a temp dir and asks
/// each LFS server for the default model's first byte or href. It never
/// downloads a model.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["JETLINK_LIVE"] == "1"))
struct LiveTests {
  @Test func theIndexAndWhereTheDefaultModelIs() async throws {
    let tmp = try TemporaryDirectory()
    let registry = Registry(layout: tmp.layout, session: .shared)

    let catalog = await registry.catalog(refresh: true)
    print("live index: \(catalog.models.count) models, error \(catalog.error ?? "none"), url \(catalog.url)")
    for model in catalog.models {
      print("  \(model.index) \(model.ref.prefix(10)) \(model.sha256?.prefix(16) ?? "-") \(model.folder ?? "") / \(model.name)")
    }
    #expect(catalog.error == nil)
    #expect(!catalog.models.isEmpty)
    #expect(catalog.models.allSatisfy { $0.sha256 != nil && $0.bytes != nil })

    let pointer = try await registry.resolve(ref: ModelIndex.defaultBigModelRef)
    print("live pointer: \(ModelIndex.defaultBigModelRef) -> \(pointer.oid) \(pointer.size) bytes")
    var found = 0
    for endpoint in LFS.endpoints {
      let href = await LFS.resolve(endpoint: endpoint, pointer: pointer, session: .shared)
      print("  \(endpoint): \(href == nil ? "no" : "yes")")
      if href != nil { found += 1 }
    }
    #expect(found > 0)
  }
}
