import Foundation
import JetlinkRegistry
import Synchronization

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// A URLSession whose requests are answered from a table, and refused when
/// they are not in it, as the Python suite's FakeOpener does. Each MockNet
/// has its own table, so tests running in parallel do not see each other's
/// routes. A table outlives its MockNet: a test that keeps only the session
/// still has its routes. Requests to 127.0.0.1 go to the real stack, for
/// LocalServer.
public final class MockNet: Sendable {
  public enum Reply: Sendable {
    case body(Data)
    /// An HTTP error status with a short body.
    case status(Int)
    /// A transport failure, as urllib's URLError.
    case failure
    /// A body delivered in pieces, for progress.
    case chunks([Data])
  }

  public struct Call: Sendable {
    public let url: String
    public let method: String
    public let body: Data?
    public let headers: [String: String]
  }

  static let header = "X-Jetlink-Test-Net"
  fileprivate static let tables = Mutex<[String: Table]>([:])
  private static let made = Mutex(0)

  fileprivate struct Table {
    var routes: [String: Reply]
    var calls: [Call] = []
  }

  let id: String
  public let session: URLSession

  public init(_ routes: [String: Reply] = [:]) {
    let configuration = URLSessionConfiguration.ephemeral
    #if canImport(FoundationNetworking)
      // swift-corelibs-foundation adds a session's extra headers only as it
      // sends, so there a request carries no sign of its session. Each net
      // gets a protocol class of its own instead, and the stack's own
      // classes after it, which serve the loopback.
      let number = MockNet.made.withLock { made in
        made += 1
        return made
      }
      id = String(number)
      configuration.protocolClasses = [MockProtocol.numbered(number)] + (configuration.protocolClasses ?? [])
    #else
      id = UUID().uuidString
      configuration.protocolClasses = [MockProtocol.self]
      configuration.httpAdditionalHeaders = [MockNet.header: id]
    #endif
    session = URLSession(configuration: configuration)
    MockNet.tables.withLock { $0[id] = Table(routes: routes) }
  }

  public subscript(url: String) -> Reply? {
    get { MockNet.tables.withLock { $0[id]?.routes[url] } }
    set { MockNet.tables.withLock { $0[id]?.routes[url] = newValue } }
  }

  public var calls: [Call] { MockNet.tables.withLock { $0[id]?.calls ?? [] } }
  public var urls: [String] { calls.map(\.url) }

  public func count(_ url: String) -> Int { urls.filter { $0 == url }.count }
}

class MockProtocol: URLProtocol, @unchecked Sendable {
  /// The table of a net whose session has this class to itself; nil for the
  /// shared class, whose requests name their table in a header.
  class var table: String? { nil }

  override class func canInit(with request: URLRequest) -> Bool {
    // Only a MockNet's session lists this class. Its requests to the local
    // server go to the real socket.
    request.url?.host != "127.0.0.1"
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    guard let client, let url = request.url, let id = request.value(forHTTPHeaderField: MockNet.header) ?? Self.table else { return }
    let body = request.httpBody ?? request.httpBodyStream.map(MockProtocol.drain)
    let headers = (request.allHTTPHeaderFields ?? [:]).filter { $0.key != MockNet.header }
    let call = MockNet.Call(url: url.absoluteString, method: request.httpMethod ?? "GET", body: body, headers: headers)
    let reply = MockNet.tables.withLock { tables -> MockNet.Reply? in
      tables[id]?.calls.append(call)
      return tables[id]?.routes[url.absoluteString]
    }
    func respond(_ status: Int, _ pieces: [Data]) {
      let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
      client.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      for piece in pieces { client.urlProtocol(self, didLoad: piece) }
      client.urlProtocolDidFinishLoading(self)
    }
    switch reply {
    case .body(let data): respond(200, [data])
    case .chunks(let pieces): respond(200, pieces)
    case .status(let status): respond(status, [Data("error \(status)".utf8)])
    case .failure, nil:
      client.urlProtocol(self, didFailWithError: URLError(.cannotFindHost, userInfo: [NSURLErrorFailingURLStringErrorKey: url.absoluteString]))
    }
  }

  override func stopLoading() {}

  private static func drain(_ stream: InputStream) -> Data {
    stream.open()
    defer { stream.close() }
    var out = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while true {
      let n = stream.read(&buffer, maxLength: buffer.count)
      if n <= 0 { break }
      out.append(buffer, count: n)
    }
    return out
  }

  /// A subclass of its own for net `number`: a type spelled from the
  /// number's bits, so each number is a distinct class without declaring one.
  static func numbered(_ number: Int) -> AnyClass {
    func tag(_ n: Int) -> any NetTag.Type {
      n == 0 ? NoTag.self : n & 1 == 1 ? one(tag(n >> 1)) : zero(tag(n >> 1))
    }
    func one<Rest: NetTag>(_: Rest.Type) -> any NetTag.Type { One<Rest>.self }
    func zero<Rest: NetTag>(_: Rest.Type) -> any NetTag.Type { Zero<Rest>.self }
    func subclass<Tag: NetTag>(_: Tag.Type) -> AnyClass { Numbered<Tag>.self }
    return subclass(tag(number))
  }
}

private final class Numbered<Tag: NetTag>: MockProtocol, @unchecked Sendable {
  override class var table: String? { String(Tag.number) }
}

protocol NetTag {
  static var number: Int { get }
}

enum NoTag: NetTag {
  static var number: Int { 0 }
}

enum Zero<Rest: NetTag>: NetTag {
  static var number: Int { Rest.number * 2 }
}

enum One<Rest: NetTag>: NetTag {
  static var number: Int { Rest.number * 2 + 1 }
}
