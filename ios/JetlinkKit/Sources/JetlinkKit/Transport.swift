// Copyright (c) 2026-, Zeph Leggett.
//
// This file is part of jetlink and is licensed under the MIT License.
// See the LICENSE file in the root directory for more details.
//
// The TCP transport and the framing over it, as jetlink/transport/tcp.py and
// StreamTransport do it.
//
// An iPhone cannot be a libusb host or a FunctionFS gadget, so the phone is
// always a TCP server: over a USB network link to the comma, or wired
// Ethernet. The framing is the Python's: a 32-byte header, the payload, and
// one pad byte when the two together are a multiple of 1024. Received
// messages are views into one reusable buffer, and a read that times out
// mid-message keeps what arrived, so a slow frame costs a frame and not the
// stream.

import Foundation

public struct LinkError: Error, CustomStringConvertible {
  public let description: String
  public let isTimeout: Bool
  init(_ d: String, timeout: Bool = false) {
    description = d
    isTimeout = timeout
  }
}

/// One received message. `payload` is valid only until the next receive on
/// the same connection.
public struct Message {
  public let header: Header
  public let payload: UnsafeRawBufferPointer

  public var type: Msg? { Msg(rawValue: header.msgType) }
  public var seq: UInt32 { header.seq }
  public var flags: Flag { Flag(rawValue: header.flags) }
  public var data: Data { Data(payload) }
}

public let defaultPort: UInt16 = 5599

private func errnoText() -> String { String(cString: strerror(errno)) }

private func setOpt(_ fd: Int32, _ level: Int32, _ name: Int32, _ value: Int32) {
  var v = value
  _ = setsockopt(fd, level, name, &v, socklen_t(MemoryLayout<Int32>.size))
}

/// Waits for `events` on fd; false on timeout. A nil timeout waits forever.
private func pollFD(_ fd: Int32, _ events: Int16, timeout: Double?) throws -> Bool {
  var p = pollfd(fd: fd, events: events, revents: 0)
  let ms: Int32 = timeout.map { Int32(max(0, min($0 * 1000, Double(Int32.max)))) } ?? -1
  while true {
    let r = poll(&p, 1, ms)
    if r > 0 { return true }
    if r == 0 { return false }
    if errno == EINTR { continue }
    throw LinkError("poll failed: \(errnoText())")
  }
}

public final class TcpListener: @unchecked Sendable {
  public let fd: Int32
  public let port: UInt16
  public let host: String

  /// Listens on host:port, IPv4 as the Python server does.
  public init(host: String = "0.0.0.0", port: UInt16 = defaultPort) throws {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 { throw LinkError("socket failed: \(errnoText())") }
    setOpt(fd, SOL_SOCKET, SO_REUSEADDR, 1)
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else {
      Darwin.close(fd)
      throw LinkError("\(host) is not an IPv4 address")
    }
    let bound = withUnsafePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    if bound != 0 {
      let e = errnoText()
      Darwin.close(fd)
      throw LinkError("cannot listen on \(host):\(port): \(e)")
    }
    if listen(fd, 1) != 0 {
      let e = errnoText()
      Darwin.close(fd)
      throw LinkError("listen failed: \(e)")
    }
    // Port 0 asks the system for one; report the one it gave.
    var actual = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafeMutablePointer(to: &actual) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
    }
    self.fd = fd
    self.port = UInt16(bigEndian: actual.sin_port)
    self.host = host
  }

  /// The next client, or nil if none arrived within `timeout`.
  public func accept(timeout: Double?) throws -> TcpConnection? {
    guard try pollFD(fd, Int16(POLLIN), timeout: timeout) else { return nil }
    var addr = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    let c = withUnsafeMutablePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.accept(fd, $0, &len) }
    }
    if c < 0 {
      if errno == EINTR || errno == EAGAIN || errno == ECONNABORTED { return nil }
      throw LinkError("accept failed: \(errnoText())")
    }
    var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
    inet_ntop(AF_INET, &addr.sin_addr, &text, socklen_t(INET_ADDRSTRLEN))
    let peer = "\(cText(text)):\(UInt16(bigEndian: addr.sin_port))"
    return TcpConnection(fd: c, peer: peer)
  }

  public func close() { Darwin.close(fd) }
}

public final class TcpConnection: @unchecked Sendable {
  public let peer: String
  private let fd: Int32
  private var closed = false
  private let closeLock = NSLock()

  // The receive buffer: [start, end) holds bytes not yet handed out.
  private var buf: UnsafeMutableRawPointer
  private var capacity: Int
  private var start = 0
  private var end = 0
  private(set) var desynced = false

  /// A send that cannot finish in this long means the peer is gone. The
  /// Python blocks for as long as TCP takes to give up; a phone should not
  /// sit on a dead link for minutes.
  public var sendTimeout: Double = 10.0
  static let readChunk = 1 << 20

  init(fd: Int32, peer: String) {
    self.fd = fd
    self.peer = peer
    capacity = 1 << 20
    buf = .allocate(byteCount: capacity, alignment: 64)
    tune()
  }

  deinit {
    close()
    buf.deallocate()
  }

  private func tune() {
    // NODELAY is the one that matters: without it the header and the body
    // can be split across a round trip.
    setOpt(fd, IPPROTO_TCP, TCP_NODELAY, 1)
    setOpt(fd, SOL_SOCKET, SO_SNDBUF, 4 << 20)
    setOpt(fd, SOL_SOCKET, SO_RCVBUF, 4 << 20)
    setOpt(fd, SOL_SOCKET, SO_KEEPALIVE, 1)
    // A cable pulled mid-drive leaves a half-open connection the next one
    // would queue behind; notice within about ten seconds.
    setOpt(fd, IPPROTO_TCP, TCP_KEEPALIVE, 4)
    setOpt(fd, IPPROTO_TCP, TCP_KEEPINTVL, 2)
    setOpt(fd, IPPROTO_TCP, TCP_KEEPCNT, 3)
    // A write to a closed socket must be an error, not a signal that kills
    // the app.
    setOpt(fd, SOL_SOCKET, SO_NOSIGPIPE, 1)
    let flags = fcntl(fd, F_GETFL)
    _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
  }

  /// A client connection, for tests and tools; the server only accepts.
  public static func connect(host: String, port: UInt16, timeout: Double = 5) throws -> TcpConnection {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 { throw LinkError("socket failed: \(errnoText())") }
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else {
      Darwin.close(fd)
      throw LinkError("\(host) is not an IPv4 address")
    }
    let rc = withUnsafePointer(to: &addr) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    if rc != 0 {
      let e = errnoText()
      Darwin.close(fd)
      throw LinkError("cannot connect to \(host):\(port): \(e)")
    }
    return TcpConnection(fd: fd, peer: "\(host):\(port)")
  }

  public func close() {
    closeLock.lock()
    defer { closeLock.unlock() }
    if !closed {
      closed = true
      Darwin.shutdown(fd, SHUT_RDWR)
      Darwin.close(fd)
    }
  }

  /// Unblocks a thread waiting in receive or send, from any thread, without
  /// releasing the descriptor under it.
  public func interrupt() {
    closeLock.lock()
    defer { closeLock.unlock() }
    if !closed { Darwin.shutdown(fd, SHUT_RDWR) }
  }

  // MARK: receiving

  private var available: Int { end - start }

  private func reserve(_ need: Int) {
    if start > 0 && start + need > capacity {
      memmove(buf, buf + start, available)
      end -= start
      start = 0
    }
    if need > capacity {
      let grown = UnsafeMutableRawPointer.allocate(byteCount: max(need, capacity * 2), alignment: 64)
      grown.copyMemory(from: buf + start, byteCount: available)
      buf.deallocate()
      buf = grown
      capacity = max(need, capacity * 2)
      end -= start
      start = 0
    }
  }

  /// Reads until `need` bytes are buffered or the deadline passes. The
  /// deadline is per message, and partial reads are kept.
  private func fill(_ need: Int, deadline: Double?) throws {
    reserve(need)
    while available < need {
      var remaining: Double?
      if let deadline {
        remaining = deadline - monotonic()
        if remaining! <= 0 {
          throw LinkError("only \(available) of \(need) bytes arrived in time", timeout: true)
        }
      }
      // Exactly what is missing, so a read never runs into the next message.
      let want = min(need - available, TcpConnection.readChunk)
      let n = Darwin.read(fd, buf + end, want)
      if n > 0 {
        end += n
        continue
      }
      if n == 0 { throw LinkError("peer closed the connection") }
      if errno == EINTR { continue }
      if errno == EAGAIN || errno == EWOULDBLOCK {
        // Wait for bytes; a wait that times out is reported by the deadline
        // check at the top of the loop.
        _ = try pollFD(fd, Int16(POLLIN), timeout: remaining)
        continue
      }
      throw LinkError("recv failed: \(errnoText())")
    }
  }

  public func receive(timeout: Double?) throws -> Message {
    if desynced { throw LinkError("stream desynced; the link must be reopened") }
    let deadline = timeout.map { monotonic() + $0 }
    try fill(Wire.headerSize, deadline: deadline)
    let header: Header
    do {
      header = try Header.decode(buf + start)
      if Int(header.length) > Wire.maxMessage {
        throw ProtocolError("message claims \(header.length) bytes, over the \(Wire.maxMessage) cap")
      }
    } catch let e as ProtocolError {
      // Nothing resynchronises a byte stream mid-message.
      desynced = true
      throw LinkError("protocol error, link unusable: \(e)")
    }
    let pad = Flag(rawValue: header.flags).contains(.padded) ? 1 : 0
    let total = Wire.headerSize + Int(header.length) + pad
    try fill(total, deadline: deadline.map { max($0, monotonic()) })
    let payload = UnsafeRawBufferPointer(start: buf + start + Wire.headerSize, count: Int(header.length))
    start += total
    if start == end {
      start = 0
      end = 0
    }
    // The payload stays where it is until the next receive reuses the space.
    return Message(header: header, payload: payload)
  }

  // MARK: sending

  private static let pad = [UInt8](repeating: 0, count: 1)

  /// Sends one message made of `parts`, header and pad added here.
  public func send(_ type: Msg, seq: UInt32, parts: [UnsafeRawBufferPointer] = [], flags: Flag = []) throws {
    var flags = flags
    let length = parts.reduce(0) { $0 + $1.count }
    var header = [UInt8](repeating: 0, count: Wire.headerSize)
    let padded = (Wire.headerSize + length) % Wire.packetMultiple == 0
    if padded { flags.insert(.padded) }
    header.withUnsafeMutableBytes {
      Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(length)).encode(into: $0.baseAddress!)
    }
    try header.withUnsafeBytes { h in
      try TcpConnection.pad.withUnsafeBytes { p in
        var all = [h] + parts.filter { $0.count > 0 }
        if padded { all.append(p) }
        try writeAll(all)
      }
    }
  }

  public func send(_ type: Msg, seq: UInt32, data: Data, flags: Flag = []) throws {
    try data.withUnsafeBytes { try send(type, seq: seq, parts: [$0], flags: flags) }
  }

  public func sendJSON(_ type: Msg, seq: UInt32, _ obj: JSON) throws {
    try send(type, seq: seq, data: obj.data)
  }

  private func writeAll(_ parts: [UnsafeRawBufferPointer]) throws {
    var iov = parts.map { iovec(iov_base: UnsafeMutableRawPointer(mutating: $0.baseAddress), iov_len: $0.count) }
    var index = 0
    let deadline = monotonic() + sendTimeout
    while index < iov.count {
      let n = iov[index...].withUnsafeBufferPointer { writev(fd, $0.baseAddress, Int32(min($0.count, Int(IOV_MAX)))) }
      if n < 0 {
        if errno == EINTR { continue }
        if errno == EAGAIN || errno == EWOULDBLOCK {
          let remaining = deadline - monotonic()
          if remaining <= 0 { throw LinkError("send timed out; link abandoned") }
          _ = try pollFD(fd, Int16(POLLOUT), timeout: remaining)
          continue
        }
        throw LinkError("send failed: \(errnoText())")
      }
      if n == 0 { throw LinkError("peer went away during send") }
      var left = n
      while left > 0 && index < iov.count {
        if left >= iov[index].iov_len {
          left -= iov[index].iov_len
          index += 1
        } else {
          iov[index].iov_base = iov[index].iov_base.map { $0 + left }
          iov[index].iov_len -= left
          left = 0
        }
      }
    }
  }
}

@inline(__always)
func monotonic() -> Double {
  Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9
}
