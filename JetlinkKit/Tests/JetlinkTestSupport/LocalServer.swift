#if canImport(Darwin) || canImport(Glibc)
  import Foundation

  #if canImport(Darwin)
    import Darwin
  #else
    import Glibc
  #endif

  /// Serves `total` bytes of a repeating pattern to every GET, over a real
  /// socket on loopback, so a download goes through URLSession's own HTTP
  /// stack. Nothing is read from disk on the serving side. `redirectURL`
  /// answers with a redirect to `url`, as comma's bucket does.
  public final class LocalServer: Sendable {
    public let port: UInt16
    public let total: Int64
    public let pattern: Data
    private let listener: Int32

    /// `total` bytes of a random pattern `patternBytes` long.
    public convenience init(total: Int64, patternBytes: Int = 1 << 20) throws {
      var generator = SystemRandomNumberGenerator()
      try self.init(pattern: Data((0..<patternBytes).map { _ in UInt8.random(in: 0...255, using: &generator) }), total: total)
    }

    /// `body`, once.
    public convenience init(serving body: Data) throws {
      try self.init(pattern: body, total: Int64(body.count))
    }

    private init(pattern: Data, total: Int64) throws {
      self.pattern = pattern
      self.total = total
      #if canImport(Darwin)
        let fd = socket(AF_INET, SOCK_STREAM, 0)
      #else
        let fd = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
      #endif
      guard fd >= 0 else { throw POSIXError(.EIO) }
      var yes: Int32 = 1
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
      var address = sockaddr_in()
      address.sin_family = sa_family_t(AF_INET)
      address.sin_addr.s_addr = inet_addr("127.0.0.1")
      address.sin_port = 0
      let bound = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
      }
      guard bound == 0, listen(fd, 8) == 0 else {
        close(fd)
        throw POSIXError(.EADDRINUSE)
      }
      var length = socklen_t(MemoryLayout<sockaddr_in>.size)
      _ = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
      }
      let port = UInt16(bigEndian: address.sin_port)
      self.port = port
      listener = fd

      Thread.detachNewThread {
        while true {
          let client = accept(fd, nil, nil)
          if client < 0 { return }  // the listener was shut down
          Thread.detachNewThread { LocalServer.serve(client, port: port, total: total, pattern: pattern) }
        }
      }
    }

    public var url: String { "http://127.0.0.1:\(port)/object" }
    public var redirectURL: String { "http://127.0.0.1:\(port)/redirect" }

    public func stop() {
      // On Linux only the shutdown wakes an accept blocked on the socket.
      shutdown(listener, Int32(SHUT_RDWR))
      close(listener)
    }

    private static func serve(_ client: Int32, port: UInt16, total: Int64, pattern: Data) {
      defer { close(client) }
      #if canImport(Darwin)
        var yes: Int32 = 1
        // A client that hangs up mid-body must not take the test process with SIGPIPE.
        setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
      #endif
      var request = Data()
      var buffer = [UInt8](repeating: 0, count: 4096)
      while request.range(of: Data("\r\n\r\n".utf8)) == nil {
        let n = read(client, &buffer, buffer.count)
        if n <= 0 { return }
        request.append(buffer, count: n)
      }
      if request.starts(with: Data("GET /redirect ".utf8)) {
        let redirect = "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:\(port)/object\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        _ = send(client, Data(redirect.utf8))
        return
      }
      let header = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: \(total)\r\nConnection: close\r\n\r\n"
      guard send(client, Data(header.utf8)) else { return }
      var left = total
      while left > 0 {
        let n = Int(min(Int64(pattern.count), left))
        guard send(client, pattern.prefix(n)) else { return }
        left -= Int64(n)
      }
    }

    private static func send(_ fd: Int32, _ data: Data) -> Bool {
      data.withUnsafeBytes { raw -> Bool in
        guard var base = raw.baseAddress else { return true }
        var left = raw.count
        while left > 0 {
          #if canImport(Darwin)
            let n = write(fd, base, left)
          #else
            // Linux has no SO_NOSIGPIPE; the flag keeps a hang-up from raising SIGPIPE.
            let n = Glibc.send(fd, base, left, Int32(MSG_NOSIGNAL))
          #endif
          if n <= 0 { return false }
          base += n
          left -= n
        }
        return true
      }
    }
  }
#endif
