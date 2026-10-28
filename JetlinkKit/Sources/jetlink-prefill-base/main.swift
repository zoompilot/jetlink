import Foundation
import JetlinkONNX

// Called by scripts/android/compile_causal_prefill.py after its source hash check.
guard CommandLine.arguments.count == 3 else {
  fatalError("usage: jetlink-prefill-base SOURCE.onnx OUTPUT_DIRECTORY")
}
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
// Fixed legacy metadata is part of the byte-verified experimental bundle.
let report = try CoreMLPreparation.prepare(
  source: source, into: directory, layout: .qnnPrefillBase,
  cacheKey: { _ in "xiaomiNativeStable1model" })
print("prepared \(report.parts.count) base graph, \(report.norms) fp16 normalizations")
