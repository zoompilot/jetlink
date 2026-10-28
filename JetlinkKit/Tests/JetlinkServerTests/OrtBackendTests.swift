import Foundation
import JetlinkKit
import JetlinkTestSupport
import Testing

@testable import JetlinkORT
@testable import JetlinkServer

/// onnxruntime's profiles: their sessions, layouts and options, which need a
/// Neural Engine or a Snapdragon to run, are only checked. The CPU profile
/// serves the golden frames in ServerTests.
@Suite("onnxruntime backend", .serialized)
struct OrtBackendTests {
  @Test("Each profile's sessions and layout")
  func profiles() {
    #if os(Android)
      let cpuLayout = "plain"
    #else
      let cpuLayout = "whole"
    #endif
    let table: [(OrtProfile, [String], [OrtUnit], String)] = [
      (.ane, ["vision", "policy"], [.coreML("CPUAndNeuralEngine"), .coreML("CPUAndGPU")], "split"),
      (.coreml, ["model"], [.coreML("CPUAndGPU")], "whole"),
      (.aneWhole, ["model"], [.coreML("ALL")], "aneWhole"),
      (.htp, ["vision", "policy"], [.htp, .qnnGPU], "split"),
      (.htpWhole, ["model"], [.htp], "aneWhole"),
      (.htpParallel, ["vision", "history", "decode", "join"], [.htp, .qnnGPU, .htp, .cpu], "qnnPrefillBase"),
      (.gpu, ["model"], [.qnnGPU], "whole"),
      (.cpu, ["model"], [.cpu], cpuLayout),
    ]
    #expect(table.map(\.0) == OrtProfile.allCases)
    for (profile, names, units, layout) in table {
      #expect(profile.sessions.map(\.name) == names, "\(profile)")
      #expect(profile.sessions.map(\.unit) == units, "\(profile)")
      #expect("\(profile.layout)" == layout, "\(profile)")
    }
    #expect(OrtProfile(rawValue: "ane-whole") == .aneWhole && OrtProfile(rawValue: "htp-whole") == .htpWhole)
    #if canImport(Metal)
      #expect(OrtProfile.available == [.ane, .aneWhole, .coreml, .cpu])
    #elseif os(Android)
      #expect(OrtProfile.available == [.htp, .htpWhole, .htpParallel, .gpu, .cpu])
    #else
      #expect(OrtProfile.available == [.cpu])
    #endif
  }

  @Test("Each unit's session plan and provider options")
  func plans() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer())
    #expect(backend.providerOptions(.htp)["backend_type"] == "htp")
    #expect(backend.providerOptions(.htp)["htp_performance_mode"] == "burst")
    #expect(backend.providerOptions(.qnnGPU)["backend_type"] == "gpu")
    let sustained = OrtBackend(profile: .htp, preparer: ONNXPreparer(), keepAlive: false)
    #expect(sustained.providerOptions(.htp)["htp_performance_mode"] == "sustained_high_performance")

    let model = URL(fileURLWithPath: "/tmp/model.onnx")
    let npu = backend.plan(.htp, model: model, cache: nil)
    #expect(npu.provider == "QNN" && npu.label == "QNN(htp)" && npu.usesNeuralEngine && !npu.usesGPU)
    let gpu = backend.plan(.qnnGPU, model: model, cache: nil)
    #expect(gpu.provider == "QNN" && gpu.label == "QNN(gpu)" && gpu.usesGPU)
    let cpu = backend.plan(.cpu, model: model, cache: nil)
    #expect(cpu.provider == nil && cpu.label == "CPU" && cpu.threads == HostChip.cpuThreads)
    let coreML = backend.plan(.coreML("ALL"), model: model, cache: URL(fileURLWithPath: "/tmp/coreml-model"))
    #expect(coreML.provider == "CoreML" && coreML.label == "CoreML(ALL)" && coreML.usesGPU && coreML.usesNeuralEngine)
    #expect(coreML.options["ModelCacheDirectory"] == "/tmp/coreml-model")
    #expect(coreML.options["SpecializationStrategy"] == "FastPrediction")
    #expect(backend.plan(.coreML("CPUAndGPU"), model: model, cache: nil).options["SpecializationStrategy"] == nil)
  }

  /// An entry names its unit as the backend that wrote it did: QNN's and the
  /// CPU's by unit, CoreML's by compute units, the CPU as null units in older
  /// Apple artifacts.
  @Test("Manifest entries, old and new, read back to their units")
  func manifestEntries() {
    for unit in [OrtUnit.coreML("CPUAndNeuralEngine"), .coreML("ALL"), .htp, .qnnGPU, .cpu] {
      #expect(OrtUnit(entry: unit.entry(model: "model.onnx", session: "model")) == unit, "\(unit)")
    }
    #expect(OrtUnit.coreML("CPUAndGPU").entry(model: "policy.onnx", session: "policy")["cache"] as? String == "coreml-policy")
    #expect(OrtUnit(entry: ["model": "model.onnx", "units": NSNull(), "cache": NSNull()]) == .cpu)
    #expect(OrtUnit(entry: ["model": "model.onnx"]) == nil)
    #expect(OrtUnit(entry: ["model": "model.onnx", "unit": "npu"]) == nil)
  }

  @Test("The chip names the artifacts")
  func chip() {
    let backend = OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "SM8650")
    #expect(backend.deviceTag() == "htp-SM8650")
    #expect(backend.tag() == "ort\(sanitize(OrtRuntime.version)).htp-SM8650")
    #expect(OrtBackend(profile: .htp, preparer: ONNXPreparer(), chip: "").deviceTag() == "htp-unknown")
    #expect(OrtBackend(profile: .cpu, preparer: ONNXPreparer()).deviceTag() == sanitize("cpu-\(HostChip.name())"))
  }
}
