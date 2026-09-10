import Foundation
import Testing

@testable import Jetlink

@Suite("Server exit messages")
struct ServerExitTests {
  @Test("A plain exit names the status and nothing else")
  func plainExit() {
    #expect(ServerStore.exitMessage(status: 1, reason: .exit) == "The server exited with status 1.")
  }

  @Test("A signal is named, and a crash points at the crash report")
  func signals() {
    let killed = ServerStore.exitMessage(status: SIGKILL, reason: .uncaughtSignal)
    #expect(killed.hasPrefix("The server was killed by signal 9 (SIGKILL)."))
    #expect(killed.contains("Crash Reports"))
    let terminated = ServerStore.exitMessage(status: SIGTERM, reason: .uncaughtSignal)
    #expect(terminated == "The server was killed by signal 15 (SIGTERM).")
    #expect(ServerStore.exitMessage(status: 40, reason: .uncaughtSignal) == "The server was killed by signal 40.")
  }

  @Test("The log tail is not in the message; the Status view shows it underneath")
  func noTail() {
    let message = ServerStore.exitMessage(status: SIGKILL, reason: .uncaughtSignal)
    #expect(!message.contains("\n"))
  }
}
