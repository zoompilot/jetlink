import Foundation

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Android)
  import Android
#endif

/// The filesystem calls the registry and the server make, with Python's
/// semantics where the port depends on them: `stat` follows a symlink as
/// `Path.stat()` does, and a file that is already gone is not an error to remove.
package enum Files {
  package struct Status {
    package let isFile: Bool
    package let isDirectory: Bool
    package let size: Int64
    package let modified: Double
  }

  /// `os.stat(path)`, or nil for anything it would raise on.
  package static func status(_ url: URL) -> Status? {
    var info = stat()
    guard stat(url.path(percentEncoded: false), &info) == 0 else { return nil }
    let type = info.st_mode & S_IFMT
    #if canImport(Darwin)
      let time = info.st_mtimespec
    #else
      let time = info.st_mtim
    #endif
    let modified = Double(time.tv_sec) + Double(time.tv_nsec) / 1e9
    return Status(isFile: type == S_IFREG, isDirectory: type == S_IFDIR, size: Int64(info.st_size), modified: modified)
  }

  static func isFile(_ url: URL) -> Bool { status(url)?.isFile ?? false }
  static func isDirectory(_ url: URL) -> Bool { status(url)?.isDirectory ?? false }
  static func exists(_ url: URL) -> Bool { status(url) != nil }

  /// The names in a directory, sorted by code point as Python sorts paths.
  /// Empty when the directory is missing, as a glob of it is.
  static func names(in directory: URL) -> [String] {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path(percentEncoded: false))) ?? []
    return names.sorted(by: codePointOrder)
  }

  static func codePointOrder(_ a: String, _ b: String) -> Bool {
    a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars)
  }

  /// A file's size, or everything under a directory artifact; 0 when neither
  /// can be read.
  package static func size(of url: URL) -> Int64 {
    guard let top = status(url) else { return 0 }
    if top.isFile { return top.size }
    guard top.isDirectory, let walker = FileManager.default.enumerator(atPath: url.path(percentEncoded: false)) else {
      return top.size
    }
    var total: Int64 = 0
    while let relative = walker.nextObject() as? String {
      if let entry = status(url.appending(path: relative)), entry.isFile {
        total += entry.size
      }
    }
    return total
  }

  /// `unlink(missing_ok=True)`: a missing file is fine, anything else throws.
  static func removeFile(_ url: URL) throws {
    guard unlink(url.path(percentEncoded: false)) == 0 || errno == ENOENT else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
  }

  /// `shutil.rmtree(path, ignore_errors=True)` for a directory, else an unlink.
  static func removeItem(_ url: URL) {
    if isDirectory(url) {
      try? FileManager.default.removeItem(at: url)
    } else {
      try? removeFile(url)
    }
  }

  static func makeDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }

  /// `os.replace`: the destination is swapped in whole or not at all.
  static func replace(_ source: URL, with destination: URL) throws {
    guard rename(source.path(percentEncoded: false), destination.path(percentEncoded: false)) == 0 else {
      throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
  }

  /// What a volume has free for a download. On iOS that is the smaller of
  /// the capacity for important usage and the plain free space: the first
  /// counts what the system could purge, and on a full iPad it stayed large
  /// while writes failed for want of space and the Neural Engine could not
  /// write its copy of a model. The plain figure is the rule elsewhere, as it
  /// is what Python's `shutil.disk_usage` reports.
  package static func freeBytes(at url: URL) -> Int64? {
    #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
      if let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey]),
        let important = values.volumeAvailableCapacityForImportantUsage
      {
        return values.volumeAvailableCapacity.map { min(important, Int64($0)) } ?? important
      }
    #endif
    if let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityKey]),
      let plain = values.volumeAvailableCapacity
    {
      return Int64(plain)
    }
    return nil
  }

  // MARK: JSON state

  /// `json.loads(path.read_text())`, or nil for anything that would raise.
  static func readJSON(_ url: URL) -> JSON? {
    guard let data = try? Data(contentsOf: url) else { return nil }
    return try? JSON.parse(data)
  }

  /// Written to a temporary file beside it and renamed into place, so a reader
  /// never sees half a file however many writers there are.
  static func writeJSON(_ value: JSON, to url: URL) throws {
    try value.data().write(to: url, options: .atomic)
  }

  // MARK: blocking work

  /// Runs file IO that takes seconds (hashing and copying a 766 MB model) on a
  /// dispatch queue, off the cooperative pool Swift's tasks share.
  static func offload<T: Sendable>(_ work: @escaping @Sendable () -> Result<T, RegistryError>) async -> Result<T, RegistryError> {
    await withCheckedContinuation { continuation in
      DispatchQueue.global(qos: .utility).async {
        continuation.resume(returning: work())
      }
    }
  }
}
