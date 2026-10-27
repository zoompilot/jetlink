import Foundation
import Testing

@testable import JetlinkORT

@Suite("QNN shared memory", .serialized)
struct OrtSharedMemoryTests {
  @Test("A CPU session cannot silently substitute heap memory for QNN RPC memory")
  func cpuDoesNotPretendToShare() throws {
    let session = try OrtSession(model: Golden("tiny_stateful").model, provider: nil)
    #expect(throws: OrtError.self) { _ = try OrtSharedAllocator(session: session) }
  }

  @Test("CPU plans cannot request HTP shared bindings")
  func wrongProvider() throws {
    let plan = SessionPlan(model: try Golden("tiny_stateful").model, provider: nil,
      options: ["enable_htp_shared_memory_allocator": "1"], label: "CPU")
    #expect(throws: OrtError.self) {
      _ = try OrtEngine(plans: [plan], device: "cpu", keepAlive: false, keepCPUWarm: false)
    }
  }
}
