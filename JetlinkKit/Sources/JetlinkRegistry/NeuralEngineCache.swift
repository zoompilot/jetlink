import Foundation

/// The Neural Engine's compiled copies of the models CoreML loads: CoreML
/// keeps them in the app's caches, under
/// `<Caches>/<bundle id>/com.apple.e5rt.e5bundlecache/<OS build>/<hash>`, one
/// per compiled model, about each model's size, and never removes a copy
/// whose model is gone. Nothing of Jetlink's names them, and the system's
/// own storage screens count them only as the app's data.
///
/// They are rebuilt when needed: CoreML compiles a model again on its next
/// load when its copy is missing (half a minute for Light Speed 2 on an iPad
/// Air (M1), a minute and a half for Lebowski). A compile is not free, so a
/// copy goes only with the artifact it was made for: the host records in each
/// artifact's sidecar the copies its build and first load added
/// (`sidecarKey`), and removing the artifact removes those and no others.
package enum NeuralEngineCache {
  static let directoryName = "com.apple.e5rt.e5bundlecache"
  /// The sidecar key that lists an artifact's copies, as `<OS build>/<hash>`.
  package static let sidecarKey = "neural_engine_bundles"

  /// Where this process's copies are: under the bundle's identifier for an
  /// app, the process's name for a command-line server.
  package static var directories: [URL] {
    #if canImport(Darwin)
      guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return [] }
      let owners = [Bundle.main.bundleIdentifier, ProcessInfo.processInfo.processName].compactMap { $0 }
      var seen = Set<String>()
      return owners.filter { seen.insert($0).inserted }.map {
        caches.appending(path: $0, directoryHint: .isDirectory).appending(path: directoryName, directoryHint: .isDirectory)
      }
    #else
      return []
    #endif
  }

  /// The copies there are, as `<OS build>/<hash>`: one per compiled model.
  package static func bundles(in directories: [URL] = directories) -> Set<String> {
    var found = Set<String>()
    for directory in directories {
      for build in Files.names(in: directory) where Files.isDirectory(directory.appending(path: build)) {
        for hash in Files.names(in: directory.appending(path: build)) where Files.isDirectory(directory.appending(path: "\(build)/\(hash)")) {
          found.insert("\(build)/\(hash)")
        }
      }
    }
    return found
  }

  /// Removes the copies `names` lists, as `bundles` names them.
  package static func remove(_ names: [String], from directories: [URL] = directories) {
    for name in names where !name.isEmpty && !name.contains("..") {
      for directory in directories {
        let bundle = directory.appending(path: name, directoryHint: .isDirectory)
        if Files.isDirectory(bundle) {
          try? FileManager.default.removeItem(at: bundle)
        }
      }
    }
  }

  /// The copies an artifact's sidecar lists.
  package static func listed(inSidecar url: URL) -> [String] {
    Files.readJSON(url)?.object?[sidecarKey]?.array?.compactMap(\.string) ?? []
  }

  /// The copies' size on disk.
  package static func bytes(in directories: [URL] = directories) -> Int64 {
    directories.filter(Files.isDirectory).reduce(0) { $0 + Files.size(of: $1) }
  }

  /// Once per caches directory, before the first build or load that records
  /// its copies: removes the copies made until then, which no sidecar lists
  /// and so nothing would ever remove (8.8 GB in 60 copies on one Mac). Each
  /// model in use compiles once more on its next load, and is listed from
  /// then on. A marker beside the copies says it is done.
  package static func forgetUntracked(in directories: [URL] = directories) {
    for directory in directories {
      let marker = directory.deletingLastPathComponent().appending(path: "\(directoryName).tracked")
      guard !Files.exists(marker) else { continue }
      try? FileManager.default.removeItem(at: directory)
      try? FileManager.default.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
      FileManager.default.createFile(atPath: marker.path(percentEncoded: false), contents: nil)
    }
  }
}
