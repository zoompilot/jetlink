import Foundation

/// A dataflow schedule over session I/O. Every layer is a completion barrier;
/// sessions in one layer read completed tensors and have disjoint outputs.
struct SessionDAG: Sendable {
  let layers: [[Int]]

  init(inputs: [[String]], outputs: [[String]]) throws {
    guard inputs.count == outputs.count, !inputs.isEmpty else { throw OrtError("empty or inconsistent session DAG") }
    var producers: [String: Int] = [:]
    for (index, names) in outputs.enumerated() {
      for name in names {
        guard producers.updateValue(index, forKey: name) == nil else { throw OrtError("multiple sessions write \(name)") }
      }
    }
    let dependencies = inputs.enumerated().map { index, names in
      Set(names.compactMap { producers[$0] }.filter { $0 != index })
    }
    var waiting = Set(inputs.indices), completed = Set<Int>(), result: [[Int]] = []
    while !waiting.isEmpty {
      let ready = waiting.filter { dependencies[$0].isSubset(of: completed) }.sorted()
      guard !ready.isEmpty else { throw OrtError("session dependency cycle") }
      result.append(ready)
      completed.formUnion(ready)
      waiting.subtract(ready)
    }
    layers = result
  }

  func run(parallel: Bool, stage: @Sendable (Int) throws -> Void) throws {
    for layer in layers {
      if parallel && layer.count > 1 {
        let failure = FirstFailure()
        // Join every branch, including after failure, before state can advance
        // or any shared tensor can be freed or reused.
        DispatchQueue.concurrentPerform(iterations: layer.count) { offset in
          do { try stage(layer[offset]) } catch { failure.store(error) }
        }
        if let error = failure.error { throw error }
      } else {
        for index in layer { try stage(index) }
      }
    }
  }
}

private final class FirstFailure: @unchecked Sendable {
  private let lock = NSLock()
  private var value: (any Error)?
  func store(_ error: any Error) {
    lock.lock()
    defer { lock.unlock() }
    if value == nil { value = error }
  }
  var error: (any Error)? {
    lock.lock()
    defer { lock.unlock() }
    return value
  }
}
