// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// Serve one client at a time, forever: _serve in jetlink/server/main.py.
//
// The engine host is shared across connections, because the comma
// reconnects at every handover and the engine must not reload. On a phone the
// listening socket can also go bad under the app (iOS reclaims the sockets of
// a suspended app), so a failed accept rebuilds the listener rather than
// ending the server.

import Foundation

public final class JetlinkServer: @unchecked Sendable {
  public let host: EngineHost
  public let bindAddress: String
  /// The port asked for; after start(), the one listened on (port 0 picks one).
  public private(set) var port: UInt16
  private let log = JLogger("jetlink.server")

  private let lock = NSLock()
  private var stopping = false
  private var running = false
  private var listener: TcpListener?
  private var connection: TcpConnection?
  private var exited = DispatchSemaphore(value: 0)

  public init(host: EngineHost, bindAddress: String = "0.0.0.0", port: UInt16 = defaultPort) {
    self.host = host
    self.bindAddress = bindAddress
    self.port = port
  }

  public var isRunning: Bool {
    lock.lock()
    defer { lock.unlock() }
    return running
  }

  /// Opens the port and starts serving on a thread of its own. Throws if the
  /// port cannot be opened, so the caller can say why.
  public func start() throws {
    lock.lock()
    defer { lock.unlock() }
    if running { return }
    let l = try TcpListener(host: bindAddress, port: port)
    listener = l
    port = l.port
    stopping = false
    running = true
    exited = DispatchSemaphore(value: 0)
    let done = exited
    let t = Thread { [self] in
      loop()
      done.signal()
    }
    t.name = "jetlink-serve"
    // The frame path runs on this thread: the queues, the engine call and the send.
    t.qualityOfService = .userInteractive
    t.stackSize = 4 << 20
    t.start()
  }

  /// Stops accepting, drops the client if there is one, and waits for the
  /// loop to finish. The engine stays loaded; `host.unload()` releases it.
  public func stop() {
    lock.lock()
    guard running else {
      lock.unlock()
      return
    }
    stopping = true
    connection?.interrupt()
    let done = exited
    lock.unlock()
    done.wait()
    lock.lock()
    listener?.close()
    listener = nil
    running = false
    lock.unlock()
  }

  private var shouldStop: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stopping
  }

  private func loop() {
    // Before accepting anything, so a client cannot race the preload.
    host.preload()
    let waiting = "listening on \(bindAddress):\(port)"
    log.info(waiting)
    host.emit("link", ["state": "waiting", "detail": .string(waiting), "peer": .null])
    while !shouldStop {
      let conn: TcpConnection
      do {
        lock.lock()
        let l = listener
        lock.unlock()
        guard let l else { break }
        guard let c = try l.accept(timeout: 0.5) else { continue }
        conn = c
      } catch {
        log.warning("the listening socket failed (\(error)); opening it again")
        reopenListener()
        continue
      }
      lock.lock()
      connection = conn
      lock.unlock()
      log.info("client connected from \(conn.peer)")
      host.emit("link", ["state": "connected", "detail": "", "peer": .string(conn.peer)])
      let session = Session(conn: conn, host: host)
      var detail = ""
      do {
        try session.serve { [self] in shouldStop }
      } catch {
        detail = String(describing: error)
        log.info("session ended: \(error)")
      }
      session.close()
      lock.lock()
      connection = nil
      lock.unlock()
      conn.close()
      host.emit("link", ["state": "disconnected", "detail": .string(detail), "peer": .null])
      log.info("client disconnected")
    }
  }

  private func reopenListener() {
    lock.lock()
    listener?.close()
    listener = nil
    lock.unlock()
    while !shouldStop {
      do {
        let l = try TcpListener(host: bindAddress, port: port)
        lock.lock()
        listener = l
        lock.unlock()
        return
      } catch {
        log.warning("cannot listen yet: \(error)")
        Thread.sleep(forTimeInterval: 1.0)
      }
    }
  }
}
