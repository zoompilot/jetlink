import Foundation
import Testing

@testable import JetlinkORT

@Suite("Fork/join session scheduling", .serialized)
struct SessionDAGTests {
  private func fork() throws -> SessionDAG {
    try SessionDAG(
      inputs: [["image"], ["vision"], ["vision"], ["off", "on"]],
      outputs: [["vision"], ["off"], ["on"], ["output"]])
  }

  @Test("Independent policies overlap, and assembly sees both complete")
  func forkJoin() throws {
    let dag = try fork()
    #expect(dag.layers == [[0], [1, 2], [3]])
    let progress = Progress()
    let offEntered = DispatchSemaphore(value: 0), onEntered = DispatchSemaphore(value: 0)
    try dag.run(parallel: true) { stage in
      switch stage {
      case 0: progress.finish(0)
      case 1:
        #expect(progress.contains(0))
        offEntered.signal()
        #expect(onEntered.wait(timeout: .now() + 2) == .success)
        progress.finish(1)
      case 2:
        #expect(progress.contains(0))
        onEntered.signal()
        #expect(offEntered.wait(timeout: .now() + 2) == .success)
        progress.finish(2)
      default:
        #expect(progress.contains(1) && progress.contains(2))
        progress.finish(3)
      }
    }
    #expect(progress.contains(3))
  }

  @Test("A failing branch is joined before the error escapes; assembly is skipped")
  func failureJoins() throws {
    let dag = try fork()
    let progress = Progress()
    let entered = DispatchSemaphore(value: 0)
    #expect(throws: Failure.self) {
      try dag.run(parallel: true) { stage in
        if stage == 1 {
          entered.signal()
          throw Failure.expected
        }
        if stage == 2 {
          #expect(entered.wait(timeout: .now() + 2) == .success)
          Thread.sleep(forTimeInterval: 0.02)
          progress.finish(2)
        }
        if stage == 3 { progress.finish(3) }
      }
    }
    #expect(progress.contains(2))
    #expect(!progress.contains(3))
  }

  @Test("The same DAG can run serially for a controlled timing comparison")
  func serialControl() throws {
    let dag = try fork()
    let progress = Progress()
    try dag.run(parallel: false) { stage in
      if stage > 0 { #expect(progress.contains(stage - 1)) }
      progress.finish(stage)
    }
    #expect(progress.contains(3))
  }

  @Test("Cycles and competing writers are rejected before sessions run")
  func rejectsInvalid() {
    #expect(throws: (any Error).self) { try SessionDAG(inputs: [["b"], ["a"]], outputs: [["a"], ["b"]]) }
    #expect(throws: (any Error).self) { try SessionDAG(inputs: [["x"], ["y"]], outputs: [["a"], ["a"]]) }
  }
}

private enum Failure: Error { case expected }
private final class Progress: @unchecked Sendable {
  private let lock = NSLock()
  private var completed = Set<Int>()
  func finish(_ stage: Int) { lock.lock(); completed.insert(stage); lock.unlock() }
  func contains(_ stage: Int) -> Bool { lock.lock(); defer { lock.unlock() }; return completed.contains(stage) }
}
