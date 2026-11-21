// An add-on over the network, where the comma's cable is one (an iPhone).
// docs/addons.md.
import Foundation

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

/// The comma listens for each add-on on its cable address, at the port its
/// ENGINE_REQ named, and the server dials it there as it dials the driving
/// model's link. An add-on the comma named no port for is never present.
public struct TCPAddonSource: AddonSource {
  static let connectTimeout: TimeInterval = 1
  let host: String
  let port: UInt16?

  public init(host: String, port: UInt16?) {
    self.host = host
    self.port = port
  }

  public func present() -> Bool {
    port != nil
  }

  public func open() throws -> AddonPipes {
    guard let port else { throw LinkError.closed("the comma named no port for this add-on") }
    return AddonPipes(SocketPipes(fd: try TCPTransport.connectSocket(host: host, port: port, timeout: Self.connectTimeout)))
  }
}

/// An add-on's byte stream on a connected socket, as `BulkPipes`.
final class SocketPipes: BulkPipes, @unchecked Sendable {
  private let fd: Int32
  private let lock = NSLock()
  private var closed = false

  init(fd: Int32) {
    self.fd = fd
    Sys.set(fd, Int32(IPPROTO_TCP), TCP_NODELAY, 1)
    #if canImport(Darwin)
      Sys.set(fd, SOL_SOCKET, SO_NOSIGPIPE, 1)
    #endif
  }

  deinit {
    close()
  }

  func read(into buffer: UnsafeMutableRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    guard try wait(Int16(POLLIN), timeout) else { return 0 }
    let got = Sys.read(fd, buffer, count)
    if got > 0 { return got }
    throw LinkError.closed(got == 0 ? "the comma closed the add-on's link" : "read: \(String(cString: strerror(errno)))")
  }

  func write(from buffer: UnsafeRawPointer, count: Int, timeout: TimeInterval) throws -> Int {
    guard try wait(Int16(POLLOUT), timeout) else { return 0 }
    var vector = iovec(iov_base: UnsafeMutableRawPointer(mutating: buffer), iov_len: count)
    let sent = Sys.writev(fd, &vector, 1)
    if sent >= 0 { return sent }
    if errno == EAGAIN || errno == EINTR { return 0 }
    throw LinkError.closed("write: \(String(cString: strerror(errno)))")
  }

  /// Whether the socket is ready for `events` within `timeout` (0: until it is).
  private func wait(_ events: Int16, _ timeout: TimeInterval) throws -> Bool {
    var poller = pollfd(fd: fd, events: events, revents: 0)
    let ready = poll(&poller, 1, timeout == 0 ? -1 : Int32(max(1, timeout * 1000)))
    if ready < 0 && errno != EINTR { throw LinkError.closed("poll: \(String(cString: strerror(errno)))") }
    if ready > 0 && poller.revents & Int16(POLLNVAL) != 0 { throw LinkError.closed("the add-on's link is closed") }
    return ready > 0
  }

  func abort() {
    lock.withLock {
      if !closed { Sys.shutdown(fd) }
    }
  }

  func close() {
    lock.withLock {
      guard !closed else { return }
      closed = true
      Sys.shutdown(fd)
      Sys.close(fd)
    }
  }
}
