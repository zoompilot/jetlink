#if os(macOS) || os(Linux)
  import Foundation
  import JetlinkRegistry
  import JetlinkServer
  import JetlinkTestSupport
  import Testing

  @testable import jetlink_server

  /// tests/test_registry.py's cli section, against `jetlink-server models`:
  /// the same subcommands, JSON and exit codes.
  struct ModelsTests {
    @Test func listJSONIsTheCatalogEvent() async throws {
      let tmp = try TemporaryDirectory()
      let run = try await models(Models.List.self, ["--json"], cache: tmp, net: MockNet(RegistryFixture.indexRoutes()))
      #expect(run.code == 0, "\(run.err)")
      let payload = try #require(try JSONSerialization.jsonObject(with: Data(run.out.utf8)) as? [String: Any])
      let rows = try #require(payload["models"] as? [[String: Any]])
      #expect(rows.count == RegistryFixture.listed && rows.first?["ref"] as? String == RegistryFixture.newestRef)
      #expect(rows.first?["sha256"] is String && rows.first?["folder"] as? String == "2026 Deep RL Models")
      #expect(payload["error"] == nil, "absent is left out")
      #expect(Set(payload.keys) == ["fetched_at", "url", "default_ref", "models"])
    }

    @Test func theListTableShowsWhatIsDownloaded() async throws {
      let tmp = try TemporaryDirectory()
      let registry = Registry(layout: CacheLayout(root: tmp.url), session: MockNet(RegistryFixture.indexRoutes()).session)
      _ = await registry.catalog()
      try Data("x".utf8).write(to: try registry.modelPath(sha256: RegistryFixture.oid))

      let run = try await models(Models.List.self, [], cache: tmp)
      #expect(run.code == 0, "\(run.err)")
      let lines = run.out.split(separator: "\n")
      #expect(lines.first == "  #  name                                         ref        built         size  state")
      let line = try #require(lines.first { $0.contains(RegistryFixture.ref.prefix(10)) })
      #expect(line.hasSuffix("downloaded") && line.contains("730 MB"), "\(line)")
      #expect(lines.filter { $0.hasSuffix("  -") }.count == RegistryFixture.listed - 1)
    }

    @Test func resolve() async throws {
      let tmp = try TemporaryDirectory()
      let net = MockNet(RegistryFixture.indexRoutes())
      let run = try await models(Models.Resolve.self, [RegistryFixture.ref, "--json"], cache: tmp, net: net)
      #expect(run.code == 0, "\(run.err)")
      let payload = try #require(try JSONSerialization.jsonObject(with: Data(run.out.utf8)) as? [String: Any])
      #expect(payload["ref"] as? String == RegistryFixture.ref && payload["sha256"] as? String == RegistryFixture.oid)
      #expect((payload["bytes"] as? NSNumber)?.int64Value == RegistryFixture.size)
      let plain = try await models(Models.Resolve.self, [RegistryFixture.ref], cache: tmp)
      #expect(plain.out == "\(RegistryFixture.oid) \(RegistryFixture.size)\n", "the index is kept: no network")
      let unlisted = try await models(Models.Resolve.self, [RegistryFixture.unlistedRef], cache: tmp)
      #expect(unlisted.out == plain.out, "a recompile is the ONNX it was built from")
      let refused = try await models(Models.Resolve.self, ["nothex"], cache: tmp)
      #expect(refused.code == 1 && refused.err == "jetlink-server: 'nothex' is not a 40 character commit\n")
    }

    @Test func rmNeedsToBeToldWhatToRemove() async throws {
      let tmp = try TemporaryDirectory()
      #expect(try await models(Models.Remove.self, [String(repeating: "b", count: 64)], cache: tmp).code == 1)
      #expect(try await models(Models.Remove.self, ["nothex", "--model"], cache: tmp).code == 1)
    }

    @Test func importThenRemove() async throws {
      let tmp = try TemporaryDirectory()
      let sha256 = try TinyModel.sha256(TinyModel.queued)
      let run = try await models(Models.Import.self, [TinyModel.queued.path, "--name", "mine"], cache: tmp)
      #expect(run.code == 0, "\(run.err)")
      let parts = run.out.split(separator: " ").map { $0.trimmingCharacters(in: .newlines) }
      #expect(parts.first == sha256)
      #expect(try Data(contentsOf: URL(fileURLWithPath: parts.last ?? "")) == Data(contentsOf: TinyModel.queued))
      #expect(run.err.hasSuffix("100% 0/0 MB\n"))

      let inventory = try await models(Models.Inventory.self, [], cache: tmp)
      #expect(inventory.out.contains("  \(sha256.prefix(16))       0 MB  mine\n"), "\(inventory.out)")

      let removed = try await models(Models.Remove.self, [sha256, "--model"], cache: tmp)
      #expect(removed.code == 0 && removed.out == "removed the model for \(sha256.prefix(16))\n")
      #expect(!FileManager.default.fileExists(atPath: parts.last ?? ""))
    }

    @Test func fetchAndInventory() async throws {
      let tmp = try TemporaryDirectory()
      let blob = try Data(contentsOf: TinyModel.queued)
      let sha256 = try TinyModel.sha256(TinyModel.queued)
      let net = MockNet(RegistryFixture.smallRoutes(blob, oid: sha256))
      let run = try await models(Models.Fetch.self, [RegistryFixture.smallRef], cache: tmp, net: net)
      #expect(run.code == 0, "\(run.err)")
      #expect(try Data(contentsOf: URL(fileURLWithPath: run.out.trimmingCharacters(in: .newlines))) == blob)

      let inventory = try await models(Models.Inventory.self, ["--json"], cache: tmp)
      let payload = try #require(try JSONSerialization.jsonObject(with: Data(inventory.out.utf8)) as? [String: Any])
      let rows = try #require(payload["models"] as? [[String: Any]])
      #expect(rows.map { $0["sha256"] as? String } == [sha256])
      #expect(payload["loaded"] == nil && payload["last_loaded"] == nil)
    }

    @Test func exitCodes() async throws {
      let tmp = try TemporaryDirectory()
      let offline = try await models(Models.Fetch.self, [RegistryFixture.smallRef], cache: tmp)
      #expect(offline.code == 2, "a network failure is 2: \(offline.err)")
      let blob = try Data(contentsOf: TinyModel.queued)
      let wrong = MockNet(RegistryFixture.smallRoutes(blob, oid: String(repeating: "b", count: 64)))
      let unverified = try await models(Models.Fetch.self, [RegistryFixture.smallRef], cache: tmp, net: wrong)
      #expect(unverified.code == 3, "bytes that do not verify are 3: \(unverified.err)")
      let unknown = try await models(Models.Fetch.self, [String(repeating: "c", count: 64)], cache: tmp, net: MockNet(RegistryFixture.indexRoutes()))
      #expect(unknown.code == 1, "a sha256 the index does not know is 1: \(unknown.err)")
    }

    @Test func prepareFetchesThenBuilds() async throws {
      let tmp = try TemporaryDirectory()
      let blob = try Data(contentsOf: TinyModel.queued)
      let sha256 = try TinyModel.sha256(TinyModel.queued)
      let net = MockNet(RegistryFixture.smallRoutes(blob, oid: sha256))
      let run = try await models(Models.Prepare.self, [RegistryFixture.smallRef, "--backend", "ort", "--device", "cpu"], cache: tmp, net: net)
      #expect(run.code == 0, "\(run.err)")
      #expect(run.err.hasPrefix("building outside the server"))
      let cache = try ServerCache(root: tmp.url, backend: try BackendOptions(device: "cpu").ort())
      #expect(try cache.entry(sha256).exists)
      #expect(cache.lastLoaded()?.sha256 == sha256)
    }
  }
#endif
