import Foundation
import Synchronization

#if canImport(CryptoKit)
  import CryptoKit
#else
  import Crypto
#endif
#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif
#if canImport(os)
  import os
#else
  import JetlinkLog
#endif

/// A large model's ONNX, named by its git-lfs oid.
public struct Pointer: Sendable, Equatable, Hashable {
  /// The ONNX's SHA-256, which is the model's identity.
  public let oid: String
  public let size: Int64

  public init(oid: String, size: Int64) {
    self.oid = oid
    self.size = size
  }

  /// `{"oid": ..., "size": ...}`, as the model index and the LFS batch API spell it.
  var json: JSON { ["oid": .string(oid), "size": .int(size)] }
}

/// Getting a large model's ONNX by its oid, the one the model index names
/// (`ModelIndex`).
///
/// The bytes are on comma's LFS servers. Since openpilot #39206 comma's own
/// LFS client stores objects in a HuggingFace bucket, read by oid; the older
/// LFS servers, and comma's model repo for the exports behind a precompiled
/// pkl, speak the LFS batch protocol, and which one has an object varies with
/// the model's age, so each is asked in turn.
///
/// This mirrors `jetlink/registry/lfs.py`: same URLs, same endpoint order,
/// same verify rules.
public enum LFS {
  /// Where comma's LFS client keeps objects since openpilot #39206. A
  /// HuggingFace bucket is not an LFS server: an object is read by its oid
  /// under it (`bucketHref`).
  public static let bucket = "https://huggingface.co/buckets/commaai/openpilot-lfs"
  public static let drivingModelsRepo = "commaai/openpilot_driving_models"
  public static let endpoints = [
    bucket,
    "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs",  // every object, older and PR-branch models included
    "https://huggingface.co/commaai/openpilot-lfs.git/info/lfs",  // the current ones
    "https://huggingface.co/\(drivingModelsRepo).git/info/lfs",  // the exports behind a precompiled pkl
  ]
  public static let mediaType = "application/vnd.git-lfs+json"
  public static let connectTimeout: TimeInterval = 30
  /// Room left on the volume beyond the model itself.
  public static let freeSlack: Int64 = 64 << 20

  /// A bucket's address for an object, which is also its download href.
  public static func bucketURL(oid: String, bucket: String = LFS.bucket) -> String {
    "\(bucket)/resolve/sha256/\(oid)"
  }

  /// A HuggingFace bucket, comma's or another, rather than an LFS server.
  static func isBucket(_ endpoint: String) -> Bool {
    endpoint.contains("/buckets/")
  }

  static let log = Logger(subsystem: "io.zoompilot.jetlink", category: "registry")

  /// A bucket's address for the object, or nil when it does not have it or
  /// cannot be reached: its first byte is asked for, as cheap as a HEAD. The
  /// bucket answers with a redirect to wherever the bytes are, which
  /// URLSession follows, here and in the download.
  static func bucketHref(
    _ bucket: String, pointer: Pointer, session: URLSession, timeout: TimeInterval = connectTimeout
  ) async -> String? {
    let url = bucketURL(oid: pointer.oid, bucket: bucket)
    do {
      _ = try await HTTP(session: session).get(url, timeout: timeout, limit: 1, headers: ["Range": "bytes=0-0"])
    } catch {
      log.warning("\(bucket, privacy: .public) has no \(pointer.oid.prefix(16), privacy: .public) (\(error.message, privacy: .public))")
      return nil
    }
    return url
  }

  // MARK: the batch API

  /// Asks one LFS server for a download href, or nil if it does not have it.
  /// A HuggingFace bucket, comma's (`bucket`) or another, is asked for the
  /// object itself.
  ///
  /// A server that is down is not different from a server that lacks the
  /// object: either way the caller moves to the next one, so nothing throws.
  static func resolve(endpoint: String, pointer: Pointer, session: URLSession, timeout: TimeInterval = connectTimeout) async -> String? {
    if isBucket(endpoint) {
      return await bucketHref(endpoint, pointer: pointer, session: session, timeout: timeout)
    }
    guard let url = URL(string: "\(endpoint)/objects/batch") else { return nil }
    let body: JSON = [
      "operation": "download",
      "transfers": ["basic"],
      "objects": [pointer.json],
    ]
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.httpBody = body.data()
    request.setValue(mediaType, forHTTPHeaderField: "Accept")
    request.setValue(mediaType, forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = timeout

    let payload: JSON
    do {
      let (data, response) = try await session.data(for: request)
      try HTTP.check(response, url: url.absoluteString)
      payload = try JSON.parse(data)
    } catch {
      log.warning("lfs batch failed at \(endpoint, privacy: .public): \(String(describing: error), privacy: .public)")
      return nil
    }

    for object in payload["objects"]?.array ?? [] {
      guard let object = object.object, object["oid"]?.string == pointer.oid else { continue }
      if object.contains("error") {
        let message = object["error"]?["message"]?.pythonString ?? "None"
        log.warning("\(endpoint, privacy: .public) has no \(pointer.oid.prefix(16), privacy: .public) (\(message, privacy: .public))")
        return nil
      }
      if let href = object["actions"]?["download"]?["href"], href.truthy {
        return href.pythonString
      }
    }
    return nil
  }
}
