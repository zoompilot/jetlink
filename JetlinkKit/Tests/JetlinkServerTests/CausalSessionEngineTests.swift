import Foundation
import JetlinkONNX
import JetlinkRegistry
import JetlinkTestSupport
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

@Suite("Same-frame recurrent session DAG", .serialized)
struct CausalSessionEngineTests {
  @Test("Real ORT bindings follow graph dependencies and preserve recurrent state", arguments: [false, true])
  func statefulBindings(parallel: Bool) throws {
    // Deliberately not topological: graph I/O determines the host interface
    // and barriers, independently of the array's order.
    let plans = ["join", "decode", "history", "vision"].map {
      SessionPlan(model: Fixture.url("parallel_\($0).onnx"), provider: nil, label: "CPU")
    }
    let engine = try OrtEngine(
      plans: plans, device: "test", keepAlive: false, keepCPUWarm: false,
      dataflow: true, parallel: parallel)
    defer { engine.close() }
    #expect(Set(engine.inputs.keys) == ["x", "state_h"])
    #expect(Set(engine.outputs.keys) == ["outputs", "next_state_h"])
    try engine.loopState([("state_h", "next_state_h")])
    let x = try #require(engine.hostInput("x"))
    let output = try #require(engine.output("outputs"))
    x.storeBytes(of: Float(2), as: Float.self)
    for frame in 0..<20 {
      try engine.run()
      #expect(output.load(as: Float.self) == Float(5 + frame))
    }
    engine.resetState()
    try engine.run()
    #expect(output.load(as: Float.self) == 5)
  }

  @Test("Generic provider discovery keeps strict accelerator admission without a chip whitelist")
  func portablePolicy() {
    for chip in ["SM8450", "SM8650", "SM8750", "future-qualcomm"] {
      let backend = OrtBackend(profile: .htpParallel, preparer: ONNXPreparer(), chip: chip)
      let model = URL(fileURLWithPath: "/unused.onnx")
      #expect(backend.qnn.strict)
      #expect(backend.deviceTag().contains(CausalPrefillBundle.identity))
      #expect(backend.deviceTag() != OrtBackend(profile: .htpWhole, preparer: ONNXPreparer(), chip: chip).deviceTag())
      #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
      #expect(backend.providerOptions(.htp)["soc_model"] == nil)
      #expect(backend.providerOptions(.htp)["htp_arch"] == nil)
      for unit in [OrtUnit.htp, .qnnGPU] {
        #expect(backend.plan(unit, model: model, cache: nil).config["session.disable_cpu_ep_fallback"] == "1")
      }
      #expect(backend.contextConfiguration(context: model)["session.disable_cpu_ep_fallback"] == "1")
    }
  }

  @Test("Unknown models choose whole NPU before bundle preparation, not after an inference error")
  func unknownModelFallback() throws {
    let backend = OrtBackend(profile: .htpParallel, preparer: FallbackPreparer())
    #expect(backend.effectiveProfile(for: CausalPrefillBundle.sourceSHA) == .htpParallel)
    #expect(backend.effectiveProfile(for: String(repeating: "a", count: 64)) == .htpWhole)
    let temporary = try TemporaryDirectory()
    #expect(throws: Intercept.whole) {
      try backend.build(
        model: Fixture.url("tiny_stateful.onnx"), artifact: temporary.url.appending(path: "model.ortcache"),
        report: { _, _, _ in }, metaExtra: [:])
    }
  }

  @Test("A cached fallback cannot be relabelled as parallel, and old route metadata rebuilds")
  func cachedRoute() throws {
    let backend = OrtBackend(profile: .htpParallel, preparer: ONNXPreparer())
    let unknown = String(repeating: "a", count: 64)
    #expect(try backend.artifactProfile(["causal_prefill_source": unknown, "causal_prefill_route": "htp-whole"]) == .htpWhole)
    #expect(try backend.artifactProfile(["causal_prefill_source": CausalPrefillBundle.sourceSHA, "causal_prefill_route": "htp-parallel"]) == .htpParallel)
    #expect(throws: ArtifactInvalid.self) {
      _ = try backend.artifactProfile(["causal_prefill_source": unknown, "causal_prefill_route": "htp-parallel"])
    }
    #expect(throws: ArtifactInvalid.self) { _ = try backend.artifactProfile([:]) }
  }
}

private enum Intercept: Error, Equatable { case whole, unexpected }
private struct FallbackPreparer: ModelPreparer {
  func readSpec(model: URL, sha256: String, nbytes: Int64, frameSkip: Int) throws -> ModelSpec { throw Intercept.unexpected }
  func prepare(model: URL, into directory: URL, layout: CoreMLPreparation.Layout, cacheKey: @escaping (String) -> String) throws -> PreparedModel {
    throw layout == .aneWhole ? Intercept.whole : Intercept.unexpected
  }
}
