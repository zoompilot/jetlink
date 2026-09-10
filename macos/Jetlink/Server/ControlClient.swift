import Foundation
import Network
import os

enum ControlClientError: Error, LocalizedError, Equatable {
  case timedOut
  case disconnected
  case notConnected
  case sendFailed(String)

  var errorDescription: String? {
    switch self {
    case .timedOut: return "The server did not answer in time."
    case .disconnected: return "The connection to the server was lost."
    case .notConnected: return "There is no connection to the server."
    case .sendFailed(let detail): return "Sending a command failed: \(detail)"
    }
  }
}

/// A client of the server's local control channel.
///
/// Marked `@unchecked Sendable` because it has mutable state: every stored
/// property other than `state` is a `let`, and every read or write of `state`
/// happens while `lock` is held, so no two threads ever see it half changed.
final class ControlClient: @unchecked Sendable {
  private final class State {
    var connection: NWConnection?
    var buffer = Data()
    var pending: [Int: CheckedContinuation<ReplyEvent, any Error>] = [:]
    var nextID = 1
    var finished = false
  }

  private let lock = NSLock()
  private let state = State()
  private let queue = DispatchQueue(label: "io.zoompilot.jetlink.control")
  private let socketPath: URL
  private let stream: AsyncStream<ControlEvent>
  private let continuation: AsyncStream<ControlEvent>.Continuation
  private let log = Logger(subsystem: "io.zoompilot.jetlink", category: "control")

  init(socketPath: URL) {
    self.socketPath = socketPath
    let (stream, continuation) = AsyncStream<ControlEvent>.makeStream(bufferingPolicy: .unbounded)
    self.stream = stream
    self.continuation = continuation
  }

  /// Every event except replies, which go to whoever sent the command.
  var events: AsyncStream<ControlEvent> { stream }

  var isConnected: Bool {
    lock.withLock { state.connection != nil && !state.finished }
  }

  // MARK: connecting

  func connect(retryingFor duration: Duration) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: duration)
    var attempts = 0
    while true {
      let remaining = clock.now.duration(to: deadline)
      if remaining <= .zero { break }
      attempts += 1
      do {
        try await attemptConnect(timeout: min(remaining, .seconds(2)))
        log.debug("connected to \(self.socketPath.path, privacy: .public) after \(attempts) attempts")
        return
      } catch {
        log.debug("connect attempt \(attempts) failed: \(error.localizedDescription, privacy: .public)")
      }
      if clock.now >= deadline { break }
      try? await Task.sleep(for: .milliseconds(250))
    }
    throw ControlClientError.timedOut
  }

  private func attemptConnect(timeout: Duration) async throws {
    let connection = NWConnection(to: .unix(path: socketPath.path(percentEncoded: false)), using: .tcp)
    let guardBox = OnceGuard()
    do {
      try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
        connection.stateUpdateHandler = { newState in
          switch newState {
          case .ready:
            if guardBox.claim() { continuation.resume() }
          case .failed(let error):
            if guardBox.claim() { continuation.resume(throwing: error) }
          case .waiting(let error):
            if guardBox.claim() { continuation.resume(throwing: error) }
          case .cancelled:
            if guardBox.claim() { continuation.resume(throwing: ControlClientError.disconnected) }
          default:
            break
          }
        }
        connection.start(queue: queue)
        queue.asyncAfter(deadline: .now() + timeout.seconds) {
          if guardBox.claim() { continuation.resume(throwing: ControlClientError.timedOut) }
        }
      }
    } catch {
      connection.cancel()
      throw error
    }

    lock.withLock {
      state.connection = connection
      state.buffer = Data()
      state.finished = false
    }
    connection.stateUpdateHandler = { [self] newState in
      switch newState {
      case .failed(let error): finish(error)
      case .cancelled: finish(ControlClientError.disconnected)
      default: break
      }
    }
    receiveLoop(connection)
  }

  // MARK: receiving

  private func receiveLoop(_ connection: NWConnection) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [self] data, _, isComplete, error in
      if let data, !data.isEmpty { ingest(data) }
      if let error {
        finish(error)
        return
      }
      if isComplete {
        finish(ControlClientError.disconnected)
        return
      }
      receiveLoop(connection)
    }
  }

  private func ingest(_ data: Data) {
    let lines: [Data] = lock.withLock {
      state.buffer.append(data)
      var out: [Data] = []
      while let index = state.buffer.firstIndex(of: 0x0A) {
        let line = state.buffer[state.buffer.startIndex..<index]
        state.buffer = state.buffer[state.buffer.index(after: index)...]
        if !line.isEmpty { out.append(Data(line)) }
      }
      return out
    }
    for line in lines { dispatch(line) }
  }

  private func dispatch(_ line: Data) {
    let event: ControlEvent
    do {
      event = try ControlEvent(jsonLine: line)
    } catch {
      log.error("dropping an undecodable control line: \(error.localizedDescription, privacy: .public)")
      return
    }
    if let reply = event.replyEvent {
      guard let id = reply.id else {
        log.error("reply with no id: \(reply.error ?? "", privacy: .public)")
        return
      }
      let waiter = lock.withLock { state.pending.removeValue(forKey: id) }
      if let waiter {
        waiter.resume(returning: reply)
      } else {
        log.debug("reply \(id) had no waiter")
      }
      return
    }
    continuation.yield(event)
  }

  // MARK: sending

  func send(_ command: ControlCommand, timeout: Duration = .seconds(30)) async throws -> ReplyEvent {
    let id = lock.withLock { () -> Int in
      let next = state.nextID
      state.nextID += 1
      return next
    }
    let line = command.jsonLine(id: id)
    let timeoutTask = Task { [self] in
      try? await Task.sleep(for: timeout)
      if Task.isCancelled { return }
      failPending(id: id, error: ControlClientError.timedOut)
    }
    defer { timeoutTask.cancel() }

    return try await withCheckedThrowingContinuation { (waiter: CheckedContinuation<ReplyEvent, any Error>) in
      let connection: NWConnection? = lock.withLock {
        guard !state.finished, let connection = state.connection else { return nil }
        state.pending[id] = waiter
        return connection
      }
      guard let connection else {
        waiter.resume(throwing: ControlClientError.notConnected)
        return
      }
      connection.send(
        content: line,
        completion: .contentProcessed { [self] error in
          if let error {
            failPending(id: id, error: ControlClientError.sendFailed(error.localizedDescription))
          }
        })
    }
  }

  private func failPending(id: Int, error: any Error) {
    let waiter = lock.withLock { state.pending.removeValue(forKey: id) }
    waiter?.resume(throwing: error)
  }

  // MARK: closing

  func close() {
    finish(ControlClientError.disconnected)
  }

  private func finish(_ error: any Error) {
    let result: (connection: NWConnection?, pending: [Int: CheckedContinuation<ReplyEvent, any Error>])? = lock.withLock {
      if state.finished { return nil }
      state.finished = true
      let connection = state.connection
      let pending = state.pending
      state.connection = nil
      state.pending = [:]
      return (connection, pending)
    }
    guard let result else { return }
    result.connection?.cancel()
    for waiter in result.pending.values { waiter.resume(throwing: error) }
    continuation.finish()
  }
}

/// A one shot latch, so a Network.framework handler that fires more than once
/// resumes its continuation exactly once. The flag is only touched under `lock`.
private final class OnceGuard: @unchecked Sendable {
  private let lock = NSLock()
  private var claimed = false

  func claim() -> Bool {
    lock.withLock {
      if claimed { return false }
      claimed = true
      return true
    }
  }
}

extension Duration {
  /// The duration as whole and fractional seconds, for APIs that want a `TimeInterval`.
  var seconds: TimeInterval {
    let parts = components
    return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
  }
}
