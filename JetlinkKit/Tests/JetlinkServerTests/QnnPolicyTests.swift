import Foundation
import JetlinkKit
import JetlinkServer
import Testing

@testable import JetlinkORT

@Suite("QNN policy", .serialized)
struct QnnPolicyTests {
  private let model = URL(fileURLWithPath: "/tmp/model.onnx")
  private let runtime = String(repeating: "a", count: 64)

  @Test("Default policy preserves existing providers, session configs and cache tags")
  func defaults() throws {
    let policy = try QnnPolicy(configuration: [:])
    #expect(policy == QnnPolicy())
    for profile in OrtProfile.allCases where profile != .htpParallel {
      let backend = OrtBackend(profile: profile, preparer: ONNXPreparer(), chip: "example", qnn: policy)
      #expect(backend.deviceTag() == "\(profile.rawValue)-example")
      #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
      #expect(backend.providerOptions(.htp)["backend_path"] == nil)
      #expect(backend.providerOptions(.htp)["enable_htp_shared_memory_allocator"] == nil)
      #expect(backend.plan(.htp, model: model, cache: nil).config.isEmpty)
    }
  }

  @Test("Strict configuration reaches the compilation and reload session plans")
  func strictBothPhases() {
    let backend = OrtBackend(profile: .htpWhole, preparer: ONNXPreparer(), qnn: QnnPolicy(strict: true))
    let key = "session.disable_cpu_ep_fallback"
    let context = URL(fileURLWithPath: "/tmp/model_ctx.onnx")
    #expect(backend.contextConfiguration(context: context)[key] == "1")
    #expect(backend.contextConfiguration(context: context)["ep.context_enable"] == "1")
    #expect(backend.contextConfiguration(context: context)["ep.context_file_path"] == context.path)
    #expect(backend.plan(.htp, model: model, cache: nil).config == [key: "1"])
    #expect(backend.plan(.qnnGPU, model: model, cache: nil).config.isEmpty)
    #expect(backend.plan(.cpu, model: model, cache: nil).config.isEmpty)
  }

  @Test("Shared I/O belongs only to HTP, independent of the numerical workaround")
  func sharedOptions() {
    let backend = OrtBackend(profile: .htpWhole, preparer: ONNXPreparer(), qnn: QnnPolicy(sharedIO: true))
    #expect(backend.providerOptions(.htp)["enable_htp_shared_memory_allocator"] == "1")
    #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
    #expect(backend.providerOptions(.qnnGPU) == ["backend_type": "gpu"])
    #expect(backend.providerOptions(.cpu).isEmpty)
    #expect(backend.providerOptions(.coreML("ALL")).isEmpty)
    #expect(backend.plan(.htp, model: model, cache: nil).config.isEmpty)
  }

  @Test("The adapter replaces backend discovery and requires matching-runtime metadata")
  func adapter() throws {
    let policy = QnnPolicy(disableAdvancedActivationFusion: true, libraryDirectory: "/app/native", runtimeIdentity: runtime)
    try policy.validate()
    let backend = OrtBackend(profile: .htpWhole, preparer: ONNXPreparer(), qnn: policy)
    #expect(backend.providerOptions(.htp)["backend_type"] == nil)
    #expect(backend.providerOptions(.htp)["backend_path"] == "/app/native/libJetlinkQnnHtpNoFusion.so")
    #expect(throws: HostError.self) { try QnnPolicy(disableAdvancedActivationFusion: true).validate() }
    #expect(throws: HostError.self) {
      try QnnPolicy(disableAdvancedActivationFusion: true, libraryDirectory: "relative", runtimeIdentity: runtime).validate()
    }
    #expect(throws: HostError.self) {
      try QnnPolicy(disableAdvancedActivationFusion: true, libraryDirectory: "/native", runtimeIdentity: "unknown").validate()
    }
  }

  @Test("All option combinations and SDK bytes have distinct HTP cache namespaces")
  func caches() {
    var tags = Set<String>()
    for shared in [false, true] {
      for strict in [false, true] {
        for fusion in [false, true] {
          let policy = QnnPolicy(sharedIO: shared, strict: strict, disableAdvancedActivationFusion: fusion,
            libraryDirectory: "/native", runtimeIdentity: runtime)
          let backend = OrtBackend(profile: .htpWhole, preparer: ONNXPreparer(), chip: "example", qnn: policy)
          tags.insert(backend.deviceTag())
          #expect(OrtBackend(profile: .cpu, preparer: ONNXPreparer(), chip: "example", qnn: policy).deviceTag() == "cpu-example")
        }
      }
    }
    #expect(tags.count == 8)
    let first = QnnPolicy(disableAdvancedActivationFusion: true, libraryDirectory: "/native", runtimeIdentity: runtime)
    let second = QnnPolicy(disableAdvancedActivationFusion: true, libraryDirectory: "/native", runtimeIdentity: String(repeating: "b", count: 64))
    #expect(first.cacheSuffix != second.cacheSuffix)
  }

  @Test("JSON accepts actual Booleans and rejects strings and numeric lookalikes")
  func typedConfiguration() throws {
    let object = try JSONSerialization.jsonObject(with: Data("{\"qnn_shared_io\":true,\"qnn_strict\":false}".utf8)) as! [String: Any]
    #expect(try QnnPolicy(configuration: object).sharedIO)
    let invalid: [Any] = ["true", "false", 1, 0, NSNull()]
    for key in ["qnn_shared_io", "qnn_strict", "qnn_no_activation_fusion"] {
      for value in invalid {
        #expect(throws: HostError.self) { _ = try QnnPolicy(configuration: [key: value]) }
      }
    }
  }
}
