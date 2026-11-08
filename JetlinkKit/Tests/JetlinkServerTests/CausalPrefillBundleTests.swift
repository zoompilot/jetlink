import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

@Suite("Hash-bound causal bundle", .serialized)
struct CausalPrefillBundleTests {
  @Test("Missing bundles and substituted sources fail before any part is copied")
  func sourceIdentity() throws {
    let temporary = try TemporaryDirectory()
    let bundle = temporary.url.appending(path: "bundle")
    let staged = temporary.url.appending(path: "staged")
    try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
    let source = Fixture.url("tiny_stateful.onnx")
    #expect(throws: HostError.self) { _ = try CausalPrefillBundle.prepare(model: source, into: staged, bundle: nil) }
    try FileManager.default.copyItem(at: Fixture.url("causal_prefill_manifest.json"), to: bundle.appending(path: "manifest.json"))
    #expect(throws: HostError.self) { _ = try CausalPrefillBundle.prepare(model: source, into: staged, bundle: bundle) }
    #expect(try FileManager.default.contentsOfDirectory(atPath: staged.path).isEmpty)
    try Data("{}".utf8).write(to: bundle.appending(path: "manifest.json"))
    #expect(throws: HostError.self) { _ = try CausalPrefillBundle.prepare(model: source, into: staged, bundle: bundle) }
  }
}
