import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// The registry's small requests: the model index, and the first byte of an
/// object in a bucket. Each failure is a network error, a 404 included, as
/// Python's `fetch_index` raises NetworkError for any of them.
struct HTTP: Sendable {
  let session: URLSession

  /// The body at `url`, at most `limit` bytes. Past the limit the request is
  /// cancelled rather than read: the bucket's first byte is asked for with a
  /// Range, and a host that ignored it would otherwise send the whole model.
  func get(_ url: String, timeout: TimeInterval, limit: Int? = nil, headers: [String: String] = [:]) async throws(RegistryError) -> Data {
    guard let target = URL(string: url), target.scheme != nil else {
      throw .network("could not fetch \(url): unknown url type")
    }
    var request = URLRequest(url: target)
    for (field, value) in headers {
      request.setValue(value, forHTTPHeaderField: field)
    }
    // URLSession's request timeout is an idle timeout, as a socket timeout is
    // for urllib: a slow body that keeps arriving is not cut off.
    request.timeoutInterval = timeout
    do {
      guard let limit else {
        let (data, response) = try await session.data(for: request)
        try HTTP.check(response, url: url)
        return data
      }
      #if canImport(FoundationNetworking)
        // swift-corelibs-foundation has no bytes(for:), so it reads the whole
        // body and keeps the first `limit` bytes. Only the bucket's
        // one-byte Range read passes a limit; a model downloads through
        // Transfer, not here.
        let (whole, response) = try await session.data(for: request)
        try HTTP.check(response, url: url)
        return Data(whole.prefix(max(0, limit)))
      #else
        let (bytes, response) = try await session.bytes(for: request)
        defer { bytes.task.cancel() }
        try HTTP.check(response, url: url)
        var data = Data()
        data.reserveCapacity(min(limit, 64 << 10))
        if limit > 0 {
          for try await byte in bytes {
            data.append(byte)
            if data.count >= limit { break }
          }
        }
        return data
      #endif
    } catch let error as RegistryError {
      throw error
    } catch {
      throw .network("could not fetch \(url): \(HTTP.describe(error))")
    }
  }

  /// urllib raises for anything but a 2xx once redirects are followed, which
  /// URLSession has also done by now.
  static func check(_ response: URLResponse, url: String) throws(RegistryError) {
    guard let failure = statusFailure(response) else { return }
    throw .network("could not fetch \(url): \(failure.text)")
  }

  /// A status urllib would raise HTTPError for, as its text reads:
  /// "HTTP Error 404: not found". Nil for a 2xx or a response that is not HTTP.
  static func statusFailure(_ response: URLResponse) -> (status: Int, text: String)? {
    guard let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) else { return nil }
    return (http.statusCode, "HTTP Error \(http.statusCode): \(HTTPURLResponse.localizedString(forStatusCode: http.statusCode))")
  }

  static func describe(_ error: any Error) -> String {
    if let error = error as? URLError {
      return "\(error.localizedDescription) (URLError \(error.code.rawValue))"
    }
    return String(describing: error)
  }
}
