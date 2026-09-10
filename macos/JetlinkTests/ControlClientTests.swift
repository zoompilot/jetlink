import Foundation
import Testing

@testable import Jetlink

/// A tiny AF_UNIX line server: it greets a client with a few fixture lines and
/// answers every line it reads with an ok reply carrying the same id.
///
/// Marked `@unchecked Sendable` because its mutable state is the two socket
/// descriptors, and both are only read or written while `lock` is held.
final class FakeControlServer: @unchecked Sendable {
  let path: String
  private let greeting: [String]
  private let queue = DispatchQueue(label: "io.zoompilot.jetlink.tests.fakecontrol")
  private let lock = NSLock()
  private var listenFD: Int32 = -1
  private var clientFD: Int32 = -1
  private var stopping = false

  init(path: String, greeting: [String]) {
    self.path = path
    self.greeting = greeting
  }

  func start() throws {
    unlink(path)
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { throw FakeServerError.cannotCreateSocket }
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
    let bytes = Array(path.utf8)
    guard bytes.count < MemoryLayout.size(ofValue: address.sun_path) else {
      close(fd)
      throw FakeServerError.pathTooLong
    }
    withUnsafeMutableBytes(of: &address.sun_path) { raw in
      raw.copyBytes(from: bytes)
      raw[bytes.count] = 0
    }
    let bound = withUnsafePointer(to: &address) { pointer in
      pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
        bind(fd, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_un>.size))
      }
    }
    guard bound == 0, listen(fd, 4) == 0 else {
      close(fd)
      throw FakeServerError.cannotBind(errno)
    }
    lock.withLock { listenFD = fd }
    queue.async { [self] in serve(fd) }
  }

  func stop() {
    let descriptors: (Int32, Int32) = lock.withLock {
      stopping = true
      let pair = (listenFD, clientFD)
      listenFD = -1
      clientFD = -1
      return pair
    }
    if descriptors.0 >= 0 { close(descriptors.0) }
    if descriptors.1 >= 0 { close(descriptors.1) }
    unlink(path)
  }

  private func serve(_ listener: Int32) {
    let client = accept(listener, nil, nil)
    guard client >= 0 else { return }
    lock.withLock { clientFD = client }
    for line in greeting { write(line, to: client) }

    var pending = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while true {
      let count = read(client, &buffer, buffer.count)
      if count <= 0 { break }
      pending.append(contentsOf: buffer[0..<count])
      while let index = pending.firstIndex(of: 0x0A) {
        let request = Data(pending[pending.startIndex..<index])
        pending = pending[pending.index(after: index)...]
        answer(request, on: client)
      }
    }
  }

  private func answer(_ request: Data, on client: Int32) {
    let object = (try? JSONSerialization.jsonObject(with: request)) as? [String: Any]
    let id = (object?["id"] as? Int) ?? 0
    write(#"{"event":"reply","t":1.0,"id":\#(id),"ok":true}"#, to: client)
  }

  private func write(_ line: String, to descriptor: Int32) {
    let data = Array((line + "\n").utf8)
    _ = data.withUnsafeBytes { raw in
      Foundation.write(descriptor, raw.baseAddress, raw.count)
    }
  }

  enum FakeServerError: Error {
    case cannotCreateSocket
    case pathTooLong
    case cannotBind(Int32)
  }
}

struct ControlClientTests {
  private static func temporarySocketPath() -> String {
    let name = "jetlink-test-\(UUID().uuidString.prefix(8)).sock"
    return (NSTemporaryDirectory() as NSString).appendingPathComponent(name)
  }

  @Test func streamsEventsAndAnswersCommands() async throws {
    let hello = #"{"event":"hello","t":1.0,"protocol":1,"pid":1,"version":"0.2.0","python":"3.14.7","platform":"darwin","cache":"/tmp","transport":"usb","port":null}"#
    let link = #"{"event":"link","t":1.1,"state":"connected","detail":"","peer":"usb"}"#
    let server = FakeControlServer(path: ControlClientTests.temporarySocketPath(), greeting: [hello, link])
    try server.start()
    defer { server.stop() }

    let client = ControlClient(socketPath: URL(filePath: server.path))
    try await client.connect(retryingFor: .seconds(5))
    defer { client.close() }

    var iterator = client.events.makeAsyncIterator()
    let first = await iterator.next()
    let second = await iterator.next()

    guard case .hello(let helloEvent) = first else {
      Issue.record("expected the hello event first")
      return
    }
    #expect(helloEvent.protocolVersion == 1)
    guard case .link(let linkEvent) = second else {
      Issue.record("expected the link event second")
      return
    }
    #expect(linkEvent.state == .connected)
    #expect(linkEvent.peer == "usb")

    let reply = try await client.send(.status, timeout: .seconds(5))
    #expect(reply.ok)
    #expect(reply.id == 1)

    let secondReply = try await client.send(.inventory, timeout: .seconds(5))
    #expect(secondReply.id == 2)
  }

  @Test func connectFailsWhenNothingIsListening() async {
    let client = ControlClient(socketPath: URL(filePath: ControlClientTests.temporarySocketPath()))
    await #expect(throws: ControlClientError.timedOut) {
      try await client.connect(retryingFor: .milliseconds(600))
    }
  }

  @Test func sendWithoutAConnectionThrows() async {
    let client = ControlClient(socketPath: URL(filePath: ControlClientTests.temporarySocketPath()))
    await #expect(throws: ControlClientError.notConnected) {
      _ = try await client.send(.status, timeout: .seconds(1))
    }
  }
}
