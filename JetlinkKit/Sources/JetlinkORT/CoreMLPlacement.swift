import Foundation

#if canImport(CoreML)
  import CoreML
#endif

/// Where CoreML plans to run a compiled session's operations, asked once when
/// an engine is built. "CPUAndNeuralEngine" is a request: what the Neural
/// Engine's compiler does not take runs on the CPU without a word, and what it
/// takes changes with the OS. macOS 27 stopped taking the attention masks
/// and four-axis MatMuls of Cinque Terre V3's policy, which took `ane-both`
/// from 13 ms a frame to 35 on an M5 Max with nothing in the log to say why.
struct CoreMLPlacement: Equatable, Sendable {
  var operations = 0
  var neuralEngine = 0
  var gpu = 0
  var cpu = 0
  /// Operators the Neural Engine does not take at all, with how many of each.
  var refused: [String: Int] = [:]

  /// The sidecar's entry for a session.
  func entry(session: String) -> [String: Any] {
    ["session": session, "operations": operations, "neural_engine": neuralEngine, "gpu": gpu, "cpu": cpu, "refused": refused]
  }

  /// One line for the log.
  func summary(session: String, wantsNeuralEngine: Bool) -> String {
    var parts = ["\(session): \(operations) operations"]
    if neuralEngine > 0 || wantsNeuralEngine { parts.append("\(neuralEngine) on the Neural Engine") }
    if gpu > 0 { parts.append("\(gpu) on the GPU") }
    parts.append("\(cpu) on the CPU")
    var line = parts.joined(separator: ", ")
    if wantsNeuralEngine, !refused.isEmpty {
      let kinds = refused.sorted { ($1.value, $0.key) < ($0.value, $1.key) }.map { "\($0.value) \($0.key)" }
      line += "; the Neural Engine does not take \(kinds.joined(separator: ", "))"
    }
    return line
  }

  /// Every compiled model under a session's CoreML cache, summed: onnxruntime
  /// writes one for each part of the graph it hands CoreML. nil when there is
  /// none to read or CoreML cannot plan it; the build goes on without.
  static func read(cache: URL, units: String) -> CoreMLPlacement? {
    #if canImport(CoreML)
      let models = compiledModels(under: cache)
      guard !models.isEmpty else { return nil }
      var total = CoreMLPlacement()
      for model in models {
        guard let one = plan(model, units: units) else { return nil }
        total.operations += one.operations
        total.neuralEngine += one.neuralEngine
        total.gpu += one.gpu
        total.cpu += one.cpu
        total.refused.merge(one.refused, uniquingKeysWith: +)
      }
      return total
    #else
      return nil
    #endif
  }

  #if canImport(CoreML)
    static func compiledModels(under cache: URL) -> [URL] {
      guard let walk = FileManager.default.enumerator(at: cache, includingPropertiesForKeys: nil) else { return [] }
      var found: [URL] = []
      for case let url as URL in walk where url.pathExtension == "mlmodelc" {
        found.append(url)
        walk.skipDescendants()
      }
      return found.sorted { $0.path < $1.path }
    }

    /// MLComputePlan loads asynchronously; a build is a blocking call on a
    /// thread of its own, so it waits.
    private static func plan(_ model: URL, units: String) -> CoreMLPlacement? {
      final class Box: @unchecked Sendable { var value: CoreMLPlacement? }
      let box = Box()
      let done = DispatchSemaphore(value: 0)
      Task.detached {
        defer { done.signal() }
        let configuration = MLModelConfiguration()
        switch units {
        case "CPUAndNeuralEngine": configuration.computeUnits = .cpuAndNeuralEngine
        case "CPUAndGPU": configuration.computeUnits = .cpuAndGPU
        case "CPUOnly": configuration.computeUnits = .cpuOnly
        default: configuration.computeUnits = .all
        }
        guard let plan = try? await MLComputePlan.load(contentsOf: model, configuration: configuration),
          case .program(let program) = plan.modelStructure
        else { return }
        var placement = CoreMLPlacement()
        for function in program.functions.values {
          count(function.block, plan, &placement)
        }
        box.value = placement
      }
      done.wait()
      return box.value
    }

    private static func count(_ block: MLModelStructure.Program.Block, _ plan: MLComputePlan, _ placement: inout CoreMLPlacement) {
      for operation in block.operations {
        for inner in operation.blocks {
          count(inner, plan, &placement)
        }
        // A constant runs nowhere, and neither does what CoreML plans no device for.
        guard operation.operatorName != "const", let usage = plan.deviceUsage(for: operation) else { continue }
        placement.operations += 1
        switch usage.preferred {
        case .neuralEngine: placement.neuralEngine += 1
        case .gpu: placement.gpu += 1
        default: placement.cpu += 1
        }
        let takes = usage.supported.contains { if case .neuralEngine = $0 { true } else { false } }
        if !takes {
          placement.refused[operation.operatorName, default: 0] += 1
        }
      }
    }
  #endif
}
