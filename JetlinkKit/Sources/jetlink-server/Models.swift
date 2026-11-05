#if os(macOS) || os(Linux)
  import ArgumentParser
  import Foundation
  import JetlinkKit
  import JetlinkRegistry
  import JetlinkServer

  /// `jetlink-server models`: the registry from a terminal, with
  /// subcommands, flags, JSON and exit codes a script can rely on.
  /// Progress goes to stderr, so stdout stays a value a script can read.
  struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
      abstract: "Large driving models: what exists, what is here, and how to get it.",
      discussion: "Exit codes: 0 ok, 1 refused or not found, 2 a network failure, 3 bytes that did not verify.",
      subcommands: [List.self, Resolve.self, Fetch.self, Import.self, Inventory.self, Remove.self, Prepare.self])
  }

  /// Where a models command writes: `out` a line a script reads, `err`
  /// text as it is, progress included.
  struct Console: Sendable {
    let out: @Sendable (String) -> Void
    let err: @Sendable (String) -> Void

    static let standard = Console(
      out: { FileHandle.standardOutput.write(Data(($0 + "\n").utf8)) },
      err: { FileHandle.standardError.write(Data($0.utf8)) })
  }

  protocol ModelsCommand: ParsableCommand, Sendable {
    /// What goes to stderr besides the progress: warnings only, so the
    /// registry's own lines stay out of a script's way.
    static var logLevel: Log.Level { get }
    var cache: CacheArguments { get }
    func run(_ registry: Registry, _ console: Console) async throws
  }

  extension ModelsCommand {
    static var logLevel: Log.Level { .warning }

    func run() throws {
      setUpLogging(Self.logLevel)
      let registry = Registry(layout: CacheLayout(root: cache.root))
      let code = try blocking { await execute(registry, .standard) }
      if code != 0 { throw ExitCode(code) }
    }

    /// The command against `registry`, and how it ended: 0 ok, 1 refused
    /// or not found, 2 a network failure, 3 bytes that did not verify.
    func execute(_ registry: Registry, _ console: Console) async -> Int32 {
      do {
        try await run(registry, console)
        return 0
      } catch let error as RegistryError {
        console.err("jetlink-server: \(error.message)\n")
        return error.kind == .verify ? 3 : error.isNetwork ? 2 : 1
      } catch let exit as ExitCode {
        return exit.rawValue
      } catch {
        console.err("jetlink-server: \(error)\n")
        return 1
      }
    }
  }

  private let megabyte: Int64 = 1 << 20

  extension Models {
    struct List: ModelsCommand {
      static let configuration = CommandConfiguration(abstract: "The big models the model index lists.")

      @Flag(help: "Refetch the model index.")
      var refresh = false
      @Flag(help: "The list as JSON: the catalog event.")
      var json = false
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        let payload = await registry.catalog(refresh: refresh)
        if json {
          console.out(jsonText(ControlEvent.catalog(payload).payload()) ?? "{}")
          return
        }
        if let error = payload.error {
          console.err("model index refresh failed: \(error)\n")
        }
        let state = states(registry)
        console.out("\(right("#", 3))  \(left("name", 44)) \(left("ref", 10)) \(left("built", 10)) \(right("size", 7))  state")
        for model in payload.models {
          let size = model.bytes.map { $0 > 0 ? "\($0 / megabyte) MB" : "?" } ?? "?"
          console.out(
            "\(right("\(model.index)", 3))  \(left(String(model.name.prefix(44)), 44)) \(left(String(model.ref.prefix(10)), 10)) "
              + "\(left(String(model.buildTime.prefix(10)), 10)) \(right(size, 7))  \(state[model.sha256 ?? ""] ?? "-")")
        }
      }

      /// sha256 to downloaded or prepared, for the table.
      private func states(_ registry: Registry) -> [String: String] {
        let inventory = registry.inventory(artifactTag: nil, artifactSuffix: "", loaded: nil)
        var out: [String: String] = [:]
        for model in inventory.models { out[model.sha256] = "downloaded" }
        for artifact in inventory.artifacts { out[artifact.sha256] = "prepared" }
        return out
      }
    }

    struct Resolve: ModelsCommand {
      static let configuration = CommandConfiguration(abstract: "The sha256 and size the model index gives a ref.")

      @Argument(help: "A ref: the model's 40 character commit.")
      var ref: String
      @Flag(help: "As JSON: ref, sha256, bytes.")
      var json = false
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        guard CacheLayout.isRef(ref) else { throw HostError.invalid("'\(ref)' is not a 40 character commit") }
        let pointer = try await registry.resolve(ref: ref)
        console.out(json ? jsonText(["ref": ref, "sha256": pointer.oid, "bytes": pointer.size] as [String: Any]) ?? "{}" : "\(pointer.oid) \(pointer.size)")
      }
    }

    struct Fetch: ModelsCommand {
      static let configuration = CommandConfiguration(abstract: "Download a model into the cache, and print where it is.")

      @Argument(help: ArgumentHelp("A ref, or the sha256 of a model the model index knows.", valueName: "ref-or-sha256"))
      var refOrSHA256: String
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        let path = try await fetch(refOrSHA256, registry, console)
        console.out(path.path)
      }
    }

    struct Import: ModelsCommand {
      static let configuration = CommandConfiguration(abstract: "Take a model from disk into the cache, and print its sha256 and path.")

      @Argument(help: "The model's ONNX.")
      var path: String
      @Option(help: "What to call it. Default: the file's name.")
      var name: String?
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        let source = URL(fileURLWithPath: path)
        let total =
          (try? FileManager.default.attributesOfItem(atPath: source.path))
          .flatMap { $0[.type] as? FileAttributeType == .typeRegular ? ($0[.size] as? NSNumber)?.int64Value : nil } ?? 0
        let local = try await registry.importModel(at: source, name: name, progress: progressLine(total: total, console))
        console.err("\n")
        console.out("\(local.sha256) \(try registry.modelPath(sha256: local.sha256).path)")
      }
    }

    struct Inventory: ModelsCommand {
      static let configuration = CommandConfiguration(abstract: "What this cache holds.")

      @Flag(help: "The inventory as JSON.")
      var json = false
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        // `current` marks the engines built for what a server here loaded
        // last: its newest engine's tag. From the cache alone, so listing
        // never opens the GPU.
        let untagged = registry.inventory(artifactTag: nil, artifactSuffix: "", loaded: nil)
        let last = untagged.artifacts.filter { $0.sha256 == untagged.lastLoaded }.max { ($0.builtAt ?? "") < ($1.builtAt ?? "") }
        // a key is "<sha16>.<tag>"
        let payload = last.map { registry.inventory(artifactTag: String($0.key.dropFirst(17)), artifactSuffix: "", loaded: nil) } ?? untagged
        if json {
          console.out(jsonText(ControlEvent.inventory(payload).payload()) ?? "{}")
          return
        }
        console.out("models")
        for model in payload.models {
          console.out(
            "  \(model.sha256.prefix(16))  \(right("\(model.bytes / megabyte)", 6)) MB  \(model.name.flatMap { $0.isEmpty ? nil : $0 } ?? "(unknown)")")
        }
        console.out("engines")
        for artifact in payload.artifacts {
          console.out(
            "  \(artifact.sha256.prefix(16))\(artifact.current ? " *" : "  ") \(right("\(artifact.bytes / megabyte)", 6)) MB  "
              + "\(left(artifact.backend, 8)) \(artifact.device)")
        }
        let disk = payload.disk
        console.out("disk: models \(disk.modelsBytes / megabyte) MB, engines \(disk.enginesBytes / megabyte) MB, free \(disk.freeBytes / megabyte) MB")
        if let last = payload.lastLoaded {
          console.out("last loaded: \(last.prefix(16))")
        }
      }
    }

    struct Remove: ModelsCommand {
      static let configuration = CommandConfiguration(commandName: "rm", abstract: "Delete a model, its engines, or both.")

      @Argument(help: "The model's sha256.")
      var sha256: String
      @Flag(help: "Delete every engine built from it.")
      var artifacts = false
      @Flag(help: "Delete the downloaded ONNX.")
      var model = false
      @OptionGroup var cache: CacheArguments

      func run(_ registry: Registry, _ console: Console) async throws {
        guard artifacts || model else { throw HostError.invalid("say what to remove, --artifacts or --model or both") }
        guard CacheLayout.isSHA256(sha256) else { throw HostError.invalid("'\(sha256)' is not a 64 character sha256") }
        try registry.remove(sha256: sha256, artifacts: artifacts, model: model)
        let removed = [("engines", artifacts), ("the model", model)].filter(\.1).map(\.0).joined(separator: " and ")
        console.out("removed \(removed) for \(sha256.prefix(16))")
      }
    }

    struct Prepare: ModelsCommand {
      static let configuration = CommandConfiguration(
        abstract: "Download a model if it is not here, then build its engine.",
        discussion: "Stop a running server on the same cache first: both would build into it.")

      @Argument(help: ArgumentHelp("A ref, or the sha256 of a model the model index knows.", valueName: "ref-or-sha256"))
      var refOrSHA256: String
      @OptionGroup var chosen: BackendArguments
      @OptionGroup var cache: CacheArguments

      /// The build's stage lines are what shows it working.
      static var logLevel: Log.Level { .info }

      func run(_ registry: Registry, _ console: Console) async throws {
        console.err("building outside the server; stop any running jetlink-server that uses this cache first\n")
        let pointer = try await registry.pointer(for: refOrSHA256)
        let path = try await fetch(pointer, registry, console)
        try await buildEngine(model: path, sha256: pointer.oid, frameSkip: Pinned.defaultFrameSkip, backend: chosen.pick(), root: registry.layout.root)
      }
    }
  }

  /// Downloads a model if it is not here yet, with its progress on stderr.
  private func fetch(_ refOrSHA256: String, _ registry: Registry, _ console: Console) async throws -> URL {
    try await fetch(try await registry.pointer(for: refOrSHA256), registry, console)
  }

  private func fetch(_ pointer: Pointer, _ registry: Registry, _ console: Console) async throws -> URL {
    let path = try await registry.fetch(pointer: pointer, progress: progressLine(total: pointer.size, console))
    console.err("\n")
    return path
  }

  /// "\r 42% 321/730 MB" on stderr at each whole percent.
  private func progressLine(total: Int64, _ console: Console) -> @Sendable (Double) -> Void {
    let shown = Locked(-1)
    return { frac in
      let percent = Int(frac * 100)
      let changed = shown.withLock { last in
        defer { last = percent }
        return last != percent
      }
      guard changed else { return }
      let done = Int64(frac * Double(total))
      console.err("\r\(right("\(percent)", 3))% \(done / megabyte)/\(total / megabyte) MB")
    }
  }

  private func left(_ text: String, _ width: Int) -> String {
    text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
  }

  private func right(_ text: String, _ width: Int) -> String {
    text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
  }
#endif
