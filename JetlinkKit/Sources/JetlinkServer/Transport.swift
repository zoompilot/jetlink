import Foundation

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

/// The link is unusable. The comma treats this as "fall back to the small model".
public enum LinkError: Error, CustomStringConvertible {
  case closed(String)
  case timedOut(String)
  case desynced(String)

  public var description: String {
    switch self {
    case .closed(let detail), .timedOut(let detail), .desynced(let detail): return detail
    }
  }
}

/// One received message. The payload is a view into the transport's receive
/// buffer, valid only until the next `recv()`: copy anything you keep.
public struct Message {
  public let msgType: UInt16
  public let seq: UInt32
  public let flags: UInt32
  public let payload: UnsafeRawBufferPointer
  /// Did it come as datagrams (`FrameDatagrams`)? Only a frame does.
  public var viaDatagram = false
}

/// Framing over a connected TCP socket, as `StreamTransport` and `TcpTransport`
/// do it in Python.
///
/// Reads never go past the end of the current message, so every message
/// lands at the start of the receive buffer and the float32 arrays inside an
/// INFER stay aligned. The steady state neither allocates nor copies.
public final class TCPTransport: @unchecked Sendable {
  public let peer: String
  private let fd: Int32
  private let reader = FrameReader(capacity: 1 << 20)
  private let sendLock = NSLock()
  private let stateLock = NSLock()
  private var closed = false
  /// The outgoing header, and the pad byte after it, packed in place for
  /// every send; `vectors` is reused too, so a frame's reply allocates nothing.
  private let tx: UnsafeMutableRawPointer
  private var vectors: [iovec] = []
  static let maxParts = 6

  init(fd: Int32, peer: String) {
    self.fd = fd
    self.peer = peer
    self.tx = UnsafeMutableRawPointer.allocate(byteCount: Wire.headerSize + 1, alignment: 8)
    self.tx.initializeMemory(as: UInt8.self, repeating: 0, count: Wire.headerSize + 1)
    self.vectors.reserveCapacity(TCPTransport.maxParts + 2)
    TCPTransport.tune(fd)
  }

  deinit {
    close()
    tx.deallocate()
    polls.deallocate()
  }

  /// A client's end: what the comma opens, and what a phone dialing the
  /// comma opens. With a `timeout`, a connect that takes longer fails
  /// rather than sit in SYN retries for a minute.
  public static func connect(host: String, port: UInt16, timeout: TimeInterval? = nil) throws -> TCPTransport {
    TCPTransport(fd: try connectSocket(host: host, port: port, timeout: timeout), peer: "\(host):\(port)")
  }

  /// A connected socket to `host`:`port`, blocking, for `connect` and an add-on's stream.
  static func connectSocket(host: String, port: UInt16, timeout: TimeInterval?) throws -> Int32 {
    let fd = socket(AF_INET, Sys.stream, 0)
    guard fd >= 0 else { throw LinkError.closed("socket: \(String(cString: strerror(errno)))") }
    var address = sockaddr_in()
    #if canImport(Darwin)
      address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    #endif
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr(host)
    let flags = fcntl(fd, F_GETFL)
    if timeout != nil {
      _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
    }
    var connected = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Sys.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    if connected != 0, let timeout, errno == EINPROGRESS {
      var poller = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
      let ready = poll(&poller, 1, Int32(max(1, timeout * 1000)))
      if ready == 0 {
        Sys.close(fd)
        throw LinkError.timedOut("could not connect to \(host):\(port) in \(timeout) s")
      }
      var error: Int32 = 0
      var length = socklen_t(MemoryLayout<Int32>.size)
      if ready > 0 && getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) == 0 && error == 0 {
        connected = 0
      } else {
        errno = error != 0 ? error : errno
      }
    }
    guard connected == 0 else {
      let reason = String(cString: strerror(errno))
      Sys.close(fd)
      throw LinkError.closed("could not connect to \(host):\(port): \(reason)")
    }
    if timeout != nil {
      _ = fcntl(fd, F_SETFL, flags)
    }
    return fd
  }

  /// A receive timeout, so a test waiting on a reply that never comes fails
  /// instead of hanging. The server's side never sets one.
  public func setReceiveTimeout(_ seconds: TimeInterval) {
    var timeout = timeval(tv_sec: .init(seconds), tv_usec: .init((seconds - Double(Int(seconds))) * 1_000_000))
    _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
  }

  // MARK: frames as datagrams

  /// Frames over the cable as datagrams, once offered; see `FrameDatagrams`.
  private var datagrams: FrameDatagrams?
  /// The stream's socket and the datagrams' for `poll`, set up with them.
  private let polls = UnsafeMutablePointer<pollfd>.allocate(capacity: 2)

  /// A port and a new token for the comma to send frames to as datagrams,
  /// or nil off the cable: only a phone's USB link is offered them. The
  /// session's loop thread calls this, as it calls `recv()`.
  public func offerDatagrams() -> (port: UInt16, token: UInt32)? {
    guard medium == .usb || TCPTransport.datagramsAnyPeer else { return nil }
    if datagrams == nil {
      guard let local = address(getsockname), let comma = address(getpeername),
        let made = FrameDatagrams(local: local, comma: comma)
      else { return nil }
      datagrams = made
      polls[0] = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
      polls[1] = pollfd(fd: made.fd, events: Int16(POLLIN), revents: 0)
    }
    return datagrams.map { ($0.port, $0.renew()) }
  }

  /// This connection's IPv4 address at one end: `getsockname` or `getpeername`.
  private func address(_ name: (Int32, UnsafeMutablePointer<sockaddr>, UnsafeMutablePointer<socklen_t>) -> Int32) -> in_addr? {
    var address = sockaddr_in()
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let named = withUnsafeMutablePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { name(fd, $0, &length) }
    }
    return named == 0 && Int32(address.sin_family) == AF_INET ? address.sin_addr : nil
  }

  /// Any link counts as the cable for frame datagrams: for the live test
  /// against the comma's client on loopback. Never on a phone.
  static let datagramsAnyPeer = ProcessInfo.processInfo.environment["JETLINK_DATAGRAMS_ANY_PEER"] == "1"

  /// The next message from either socket. TCP first when both have one, and
  /// a TCP message is read whole once begun.
  private func pollBoth(_ datagrams: FrameDatagrams) throws -> Message? {
    while true {
      polls[0].revents = 0
      polls[1].revents = 0
      if poll(polls, 2, -1) < 0 {
        if errno == EINTR { continue }
        throw LinkError.closed("poll failed: \(String(cString: strerror(errno)))")
      }
      if polls[0].revents != 0 { return nil }
      if polls[1].revents != 0, let whole = datagrams.read(), let message = TCPTransport.message(whole) {
        return message
      }
    }
  }

  /// A datagram frame's bytes as a message, or nil for bytes that do not
  /// hold together: those are dropped, never a desynced link.
  static func message(_ bytes: UnsafeRawBufferPointer) -> Message? {
    guard let base = bytes.baseAddress, let header = try? Wire.unpackHeader(base),
      Wire.headerSize + Int(header.length) + Wire.pad(header) == bytes.count
    else { return nil }
    return Message(
      msgType: header.msgType, seq: header.seq, flags: header.flags,
      payload: UnsafeRawBufferPointer(start: base + Wire.headerSize, count: Int(header.length)), viaDatagram: true)
  }

  // MARK: receiving

  public func recv() throws -> Message {
    if let datagrams, let message = try pollBoth(datagrams) {
      return message
    }
    return try reader.recv(pad: Wire.pad) { into, missing in
      while true {
        let n = Sys.read(fd, into, missing)
        if n > 0 { return n }
        if n == 0 { throw LinkError.closed("peer closed the connection") }
        if errno == EINTR { continue }
        if errno == EAGAIN || errno == EWOULDBLOCK {
          throw LinkError.timedOut("only part of a message arrived in time")
        }
        throw LinkError.closed("recv failed: \(String(cString: strerror(errno)))")
      }
    }
  }

  // MARK: sending

  /// One message, `parts` concatenated as its payload, in one vectored write:
  /// header, parts and the pad byte together, so the kernel never sees the
  /// header as a segment of its own. `MessageLink` has the other forms.
  public func sendParts(_ type: Wire.Msg, seq: UInt32, parts: UnsafeBufferPointer<UnsafeRawBufferPointer>, flags: Wire.Flag) throws {
    sendLock.lock()
    defer { sendLock.unlock() }
    var flags = flags
    var length = 0
    for part in parts { length += part.count }
    let padded = Wire.needsPad(length)
    if padded {
      flags.insert(.padded)
    }
    Wire.packHeader(Wire.Header(msgType: type.rawValue, seq: seq, flags: flags.rawValue, length: UInt32(length)), into: tx)
    vectors.removeAll(keepingCapacity: true)
    vectors.append(iovec(iov_base: tx, iov_len: numericCast(Wire.headerSize)))
    for part in parts where part.count > 0 {
      vectors.append(iovec(iov_base: UnsafeMutableRawPointer(mutating: part.baseAddress), iov_len: numericCast(part.count)))
    }
    if padded {
      vectors.append(iovec(iov_base: tx + Wire.headerSize, iov_len: 1))
    }
    try writeAll(&vectors)
  }

  private func writeAll(_ vectors: inout [iovec]) throws {
    var index = 0
    while index < vectors.count {
      let n = vectors.withUnsafeBufferPointer { buffer in
        Sys.writev(fd, buffer.baseAddress! + index, min(vectors.count - index, Sys.iovMax))
      }
      if n < 0 {
        if errno == EINTR { continue }
        if errno == EAGAIN || errno == EWOULDBLOCK {
          throw LinkError.timedOut("send timed out; link abandoned")
        }
        throw LinkError.closed("send failed: \(String(cString: strerror(errno)))")
      }
      if n == 0 {
        throw LinkError.closed("peer went away during send")
      }
      var remaining = n
      while remaining > 0 && index < vectors.count {
        // iov_len is size_t: Int on glibc, UInt on Bionic.
        let length = Int(vectors[index].iov_len)
        if remaining >= length {
          remaining -= length
          index += 1
        } else {
          vectors[index].iov_base = vectors[index].iov_base.map { $0 + remaining }
          vectors[index].iov_len = numericCast(length - remaining)
          remaining = 0
        }
      }
    }
  }

  // MARK: lifecycle

  /// Wakes a `recv()` blocked on another thread; it then throws `.closed`.
  public func shutdown() {
    stateLock.lock()
    defer { stateLock.unlock() }
    if !closed {
      _ = Sys.shutdown(fd)
    }
  }

  public func close() {
    stateLock.lock()
    defer { stateLock.unlock() }
    if !closed {
      closed = true
      _ = Sys.shutdown(fd)
      _ = Sys.close(fd)
      datagrams?.close()
    }
  }

  /// NODELAY is the one that matters: without it the header and the body can be
  /// split across an RTT. The keepalive is short because a pulled cable
  /// otherwise leaves the session blocked in recv for hours, and the comma's
  /// reconnect waiting behind it.
  private static func tune(_ fd: Int32) {
    Sys.tune(fd, keepalive: (idle: 5, interval: 2, count: 3), sendTimeout: 10)
    Sys.set(fd, SOL_SOCKET, SO_SNDBUF, 4 << 20)
    Sys.set(fd, SOL_SOCKET, SO_RCVBUF, 4 << 20)
  }
}

/// A listening TCP socket. `accept` polls so `close` can stop it from another
/// thread, and frees the port at once.
public final class TCPListener: @unchecked Sendable {
  public let port: UInt16
  private let fd: Int32
  private let lock = NSLock()
  private var closed = false

  /// Listens on `host`, or with `dualStack` on every address: IPv6 with IPv4
  /// mapped in, since a phone may resolve a name to either, else IPv4 on `host`.
  public init(host: String = "0.0.0.0", port: UInt16 = Wire.defaultPort, dualStack: Bool = false) throws {
    var six = sockaddr_in6()
    #if canImport(Darwin)
      six.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
    #endif
    six.sin6_family = sa_family_t(AF_INET6)
    six.sin6_port = port.bigEndian
    var four = sockaddr_in()
    #if canImport(Darwin)
      four.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    #endif
    four.sin_family = sa_family_t(AF_INET)
    four.sin_port = port.bigEndian
    four.sin_addr.s_addr = inet_addr(host)
    let bound = (dualStack ? TCPListener.listen(&six, AF_INET6) : nil) ?? TCPListener.listen(&four, AF_INET)
    guard let bound else {
      throw LinkError.closed("could not listen on \(dualStack ? "port" : "\(host):")\(port): \(String(cString: strerror(errno)))")
    }
    // `accept` runs under the lock, so it must never wait there.
    _ = fcntl(bound, F_SETFL, fcntl(bound, F_GETFL) | O_NONBLOCK)
    var actual = sockaddr_storage()
    var length = socklen_t(MemoryLayout<sockaddr_storage>.size)
    _ = withUnsafeMutablePointer(to: &actual) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(bound, $0, &length) }
    }
    fd = bound
    self.port = TCPListener.peer(actual).port
  }

  /// A socket bound to `address` and listening, or nil with errno set.
  private static func listen<Address>(_ address: inout Address, _ family: Int32) -> Int32? {
    let fd = socket(family, Sys.stream, 0)
    guard fd >= 0 else { return nil }
    Sys.set(fd, SOL_SOCKET, SO_REUSEADDR, 1)
    if family == AF_INET6 {
      Sys.set(fd, Int32(IPPROTO_IPV6), IPV6_V6ONLY, 0)
    }
    let bound = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<Address>.size)) }
    }
    guard bound == 0, Sys.listen(fd, 16) == 0 else {
      let failure = errno
      Sys.close(fd)
      errno = failure
      return nil
    }
    return fd
  }

  deinit {
    close()
  }

  /// The next client, or nil once the listener is closed. Blocks.
  public func accept() -> TCPTransport? {
    acceptConnection().map { TCPTransport(fd: $0.fd, peer: $0.peer) }
  }

  /// The next client's descriptor, blocking, and its address; the caller
  /// closes the descriptor. Nil once the listener is closed.
  /// A client's socket, `host:port`, and the host alone.
  package func acceptConnection() -> (fd: Int32, peer: String, host: String)? {
    while true {
      lock.lock()
      let isClosed = closed
      lock.unlock()
      if isClosed { return nil }
      var poller = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
      let ready = poll(&poller, 1, 250)
      if ready < 0 && errno != EINTR { return nil }
      if ready <= 0 { continue }
      // iOS reclaims a suspended app's listening sockets; this one is gone.
      if poller.revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 { return nil }
      var address = sockaddr_storage()
      var length = socklen_t(MemoryLayout<sockaddr_storage>.size)
      // Under the lock, and only while open: a `close` during the poll frees
      // the descriptor, and the next listener can get its number, whose
      // clients this one would otherwise take.
      lock.lock()
      if closed {
        lock.unlock()
        return nil
      }
      let client = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Sys.accept(fd, $0, &length) }
      }
      let failure = errno
      lock.unlock()
      if client < 0 {
        if failure == EINTR || failure == ECONNABORTED || failure == EAGAIN || failure == EWOULDBLOCK { continue }
        return nil
      }
      // BSD hands the listener's O_NONBLOCK on to what it accepts.
      _ = fcntl(client, F_SETFL, fcntl(client, F_GETFL) & ~O_NONBLOCK)
      let (host, port) = TCPListener.peer(address)
      return (client, "\(host):\(port)", host)
    }
  }

  /// A socket address's numeric host and port.
  private static func peer(_ address: sockaddr_storage) -> (host: String, port: UInt16) {
    var address = address
    var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
    var service = [CChar](repeating: 0, count: Int(NI_MAXSERV))
    let length = socklen_t(Int32(address.ss_family) == AF_INET6 ? MemoryLayout<sockaddr_in6>.size : MemoryLayout<sockaddr_in>.size)
    _ = withUnsafePointer(to: &address) {
      $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        getnameinfo($0, length, &host, .init(host.count), &service, .init(service.count), NI_NUMERICHOST | NI_NUMERICSERV)
      }
    }
    return (String(cString: host), UInt16(String(cString: service)) ?? 0)
  }

  public func close() {
    lock.lock()
    defer { lock.unlock() }
    if !closed {
      closed = true
      // Linux keeps a socket that another thread polls bound until the poll
      // returns, so a restart's bind would fail for up to a poll's 250 ms.
      // A shutdown takes it out of LISTEN and wakes the poll.
      _ = Sys.shutdown(fd)
      Sys.close(fd)
    }
  }
}

/// JSON the way the Python server writes it, for payloads and control lines.
enum JSONLine {
  static func encode(_ object: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data("{}".utf8)
  }

  static func decode(_ bytes: UnsafeRawBufferPointer) -> [String: Any]? {
    guard bytes.count > 0 else { return [:] }
    let data = Data(bytes)
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
  }
}

/// The socket calls whose names the transports' own methods shadow, and the
/// few constants that differ between Darwin, glibc (Linux) and Bionic
/// (Android). The status page's server uses them too.
package enum Sys {
  #if canImport(Darwin)
    package static let stream = SOCK_STREAM
    static let datagram = SOCK_DGRAM
    static let keepIdle = TCP_KEEPALIVE
    static let iovMax = Int(IOV_MAX)
    /// For `send`: Darwin sockets have SO_NOSIGPIPE set instead.
    package static let sendFlags: Int32 = 0
  #elseif canImport(Glibc)
    package static let stream = Int32(SOCK_STREAM.rawValue)
    static let datagram = Int32(SOCK_DGRAM.rawValue)
    static let keepIdle = TCP_KEEPIDLE
    static let iovMax = 1024
    package static let sendFlags = Int32(MSG_NOSIGNAL)
  #else
    package static let stream = SOCK_STREAM
    static let datagram = SOCK_DGRAM
    static let keepIdle = TCP_KEEPIDLE
    static let iovMax = 1024
    package static let sendFlags = Int32(MSG_NOSIGNAL)
  #endif

  /// An integer socket option.
  package static func set(_ fd: Int32, _ level: Int32, _ option: Int32, _ value: Int32) {
    var value = value
    _ = setsockopt(fd, level, option, &value, socklen_t(MemoryLayout<Int32>.size))
  }

  /// A connected socket: NODELAY, keepalive probes after `idle` seconds of
  /// silence, `interval` apart, `count` of them, a send that gives up after
  /// `sendTimeout` without progress, and no SIGPIPE.
  package static func tune(_ fd: Int32, keepalive: (idle: Int32, interval: Int32, count: Int32), sendTimeout: TimeInterval) {
    set(fd, Int32(IPPROTO_TCP), TCP_NODELAY, 1)
    #if canImport(Darwin)
      // Linux has no such option; `writev` and `sendFlags` say MSG_NOSIGNAL there.
      set(fd, SOL_SOCKET, SO_NOSIGPIPE, 1)
    #endif
    set(fd, SOL_SOCKET, SO_KEEPALIVE, 1)
    set(fd, Int32(IPPROTO_TCP), keepIdle, keepalive.idle)
    set(fd, Int32(IPPROTO_TCP), TCP_KEEPINTVL, keepalive.interval)
    set(fd, Int32(IPPROTO_TCP), TCP_KEEPCNT, keepalive.count)
    var timeout = timeval(tv_sec: .init(sendTimeout), tv_usec: .init((sendTimeout - sendTimeout.rounded(.down)) * 1_000_000))
    _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
  }

  static func listen(_ fd: Int32, _ backlog: Int32) -> Int32 {
    #if canImport(Darwin)
      Darwin.listen(fd, backlog)
    #elseif canImport(Glibc)
      Glibc.listen(fd, backlog)
    #else
      Android.listen(fd, backlog)
    #endif
  }

  @discardableResult
  package static func close(_ fd: Int32) -> Int32 {
    #if canImport(Darwin)
      Darwin.close(fd)
    #elseif canImport(Glibc)
      Glibc.close(fd)
    #else
      Android.close(fd)
    #endif
  }

  @discardableResult
  package static func shutdown(_ fd: Int32) -> Int32 {
    #if canImport(Darwin)
      Darwin.shutdown(fd, SHUT_RDWR)
    #elseif canImport(Glibc)
      Glibc.shutdown(fd, Int32(SHUT_RDWR))
    #else
      Android.shutdown(fd, Int32(SHUT_RDWR))
    #endif
  }

  static func read(_ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int) -> Int {
    #if canImport(Darwin)
      Darwin.read(fd, buffer, count)
    #elseif canImport(Glibc)
      Glibc.read(fd, buffer, count)
    #else
      Android.read(fd, buffer, count)
    #endif
  }

  /// A vectored write that is an error, never SIGPIPE, on a closed socket.
  static func writev(_ fd: Int32, _ vectors: UnsafePointer<iovec>, _ count: Int) -> Int {
    #if canImport(Darwin)
      Darwin.writev(fd, vectors, Int32(count))
    #elseif canImport(Glibc)
      var message = msghdr()
      message.msg_iov = UnsafeMutablePointer(mutating: vectors)
      message.msg_iovlen = count
      return Glibc.sendmsg(fd, &message, Int32(MSG_NOSIGNAL))
    #else
      var message = msghdr()
      message.msg_iov = UnsafeMutablePointer(mutating: vectors)
      message.msg_iovlen = count
      return Android.sendmsg(fd, &message, MSG_NOSIGNAL)
    #endif
  }

  static func connect(_ fd: Int32, _ address: UnsafePointer<sockaddr>, _ length: socklen_t) -> Int32 {
    #if canImport(Darwin)
      Darwin.connect(fd, address, length)
    #elseif canImport(Glibc)
      Glibc.connect(fd, address, length)
    #else
      Android.connect(fd, address, length)
    #endif
  }

  static func accept(_ fd: Int32, _ address: UnsafeMutablePointer<sockaddr>, _ length: UnsafeMutablePointer<socklen_t>) -> Int32 {
    #if canImport(Darwin)
      Darwin.accept(fd, address, length)
    #elseif canImport(Glibc)
      Glibc.accept(fd, address, length)
    #else
      Android.accept(fd, address, length)
    #endif
  }
}
