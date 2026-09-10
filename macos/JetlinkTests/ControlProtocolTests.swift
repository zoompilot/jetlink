import Foundation
import Testing

@testable import Jetlink

/// The fixture files live next to this source file, so both SwiftPM and Xcode
/// find them without a resource bundle.
enum Fixture {
  static let directory = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "Fixtures")

  static func data(_ name: String) throws -> Data {
    try Data(contentsOf: directory.appending(path: name))
  }

  static func lines(_ name: String) throws -> [Data] {
    let text = try String(decoding: data(name), as: UTF8.self)
    return text.split(separator: "\n", omittingEmptySubsequences: true).map { Data($0.utf8) }
  }
}

struct ControlProtocolTests {
  private func events() throws -> [Data] {
    try Fixture.lines("control_events.jsonl")
  }

  @Test func fixtureHasEveryEventKind() throws {
    #expect(try events().count == 18)
  }

  @Test func decodesHello() throws {
    guard case .hello(let hello) = try ControlEvent(jsonLine: events()[0]) else {
      Issue.record("expected a hello event")
      return
    }
    #expect(hello.protocolVersion == 1)
    #expect(hello.pid == 4242)
    #expect(hello.version == "0.2.0")
    #expect(hello.python == "3.14.7")
    #expect(hello.platform == "darwin")
    #expect(hello.cache == "/Users/me/Library/Application Support/Jetlink/cache")
    #expect(hello.transport == "usb")
    #expect(hello.port == nil)
  }

  @Test func decodesServer() throws {
    guard case .server(let server) = try ControlEvent(jsonLine: events()[1]) else {
      Issue.record("expected a server event")
      return
    }
    #expect(server.state == "serving")
    #expect(server.detail.isEmpty)
    #expect(server.backend == "ort")
    #expect(server.runtimeVersion == "1.29.0")
    #expect(server.device == "coreml-Apple_M1_Pro")
  }

  @Test func decodesLink() throws {
    guard case .link(let waiting) = try ControlEvent(jsonLine: events()[2]),
          case .link(let connected) = try ControlEvent(jsonLine: events()[3]) else {
      Issue.record("expected two link events")
      return
    }
    #expect(waiting.state == .waiting)
    #expect(waiting.detail == "waiting for a jetlink gadget at 1209:0001")
    #expect(waiting.peer == nil)
    #expect(connected.state == .connected)
    #expect(connected.peer == "usb")
  }

  @Test func decodesEngine() throws {
    guard case .engine(let idle) = try ControlEvent(jsonLine: events()[4]),
          case .engine(let building) = try ControlEvent(jsonLine: events()[5]),
          case .engine(let ready) = try ControlEvent(jsonLine: events()[6]) else {
      Issue.record("expected three engine events")
      return
    }
    #expect(idle.state == .none)
    #expect(idle.sha256 == nil)
    #expect(idle.stage == nil)
    #expect(idle.loadOnly == false)
    #expect(building.state == .building)
    #expect(building.stage == "build")
    #expect(building.frac == 0.42)
    #expect(building.msg == "compiling the graph")
    #expect(ready.state == .ready)
    #expect(ready.loadOnly == true)
    #expect(ready.sha256 == "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525")
  }

  @Test func decodesStats() throws {
    guard case .stats(let stats) = try ControlEvent(jsonLine: events()[7]) else {
      Issue.record("expected a stats event")
      return
    }
    #expect(stats.frames == 1234)
    #expect(stats.fps == 19.9)
    #expect(stats.totalMs.mean == 31.2)
    #expect(stats.totalMs.p99 == 38.0)
    #expect(stats.totalMs.max == 41.5)
    #expect(stats.gpuMs.mean == 21.0)
    #expect(stats.slow == 0)
    #expect(stats.windowS == 1.0)
  }

  @Test func decodesInventory() throws {
    guard case .inventory(let inventory) = try ControlEvent(jsonLine: events()[8]) else {
      Issue.record("expected an inventory event")
      return
    }
    #expect(inventory.loaded == "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525")
    #expect(inventory.lastLoaded == inventory.loaded)
    #expect(inventory.models.count == 1)
    #expect(inventory.models[0].bytes == 765_953_504)
    #expect(inventory.models[0].name == "BMRLNAP Model v4 (August 30, 2026)")
    #expect(inventory.models[0].ref == "f877d7a0ccc3cce943c76e285214c020cd65c899")
    #expect(inventory.models[0].id == inventory.models[0].sha256)
    #expect(inventory.artifacts.count == 1)
    let artifact = inventory.artifacts[0]
    #expect(artifact.key == "a086d5249fc308bb.ort1.29.0.coreml-Apple_M1_Pro")
    #expect(artifact.id == artifact.key)
    #expect(artifact.bytes == 5_900_000_000)
    #expect(artifact.backend == "ort")
    #expect(artifact.runtimeVersion == "1.29.0")
    #expect(artifact.device == "coreml-Apple_M1_Pro")
    #expect(artifact.builtAt == "2026-09-08T21:19:15Z")
    #expect(artifact.buildSeconds == 548.9)
    #expect(artifact.checkpoint == "b9facbcc-3a1e-4d3f-9a55-6d1a0f2c8e77")
    #expect(artifact.current)
    #expect(inventory.disk.modelsBytes == 765_953_504)
    #expect(inventory.disk.enginesBytes == 5_900_000_000)
    #expect(inventory.disk.freeBytes == 120_000_000_000)
  }

  @Test func decodesCatalog() throws {
    guard case .catalog(let catalog) = try ControlEvent(jsonLine: events()[9]) else {
      Issue.record("expected a catalog event")
      return
    }
    #expect(catalog.fetchedAt == 1_757_440_000.0)
    #expect(catalog.defaultRef == "f877d7a0ccc3cce943c76e285214c020cd65c899")
    #expect(catalog.error == nil)
    #expect(catalog.models.count == 2)
    #expect(catalog.models[0].shortName == "CTMV2")
    #expect(catalog.models[0].index == 12)
    #expect(catalog.models[0].sha256 == nil)
    #expect(catalog.models[0].bytes == nil)
    #expect(catalog.models[0].id == catalog.models[0].ref)
    #expect(catalog.models[1].bytes == 765_953_504)
    #expect(catalog.models[1].buildTime == "2026-09-01T05:39:17Z")
  }

  @Test func decodesDownload() throws {
    guard case .download(let download) = try ControlEvent(jsonLine: events()[10]) else {
      Issue.record("expected a download event")
      return
    }
    #expect(download.state == "progress")
    #expect(download.frac == 0.42)
    #expect(download.bytes == 321_000_000)
    #expect(download.total == 765_953_504)
    #expect(download.rateBps == 41_000_000.0)
    #expect(download.ref == "f877d7a0ccc3cce943c76e285214c020cd65c899")
    #expect(download.source == "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs")
  }

  @Test func decodesImport() throws {
    guard case .importEvent(let value) = try ControlEvent(jsonLine: events()[11]) else {
      Issue.record("expected an import event")
      return
    }
    #expect(value.path == "/Users/me/Downloads/big.onnx")
    #expect(value.state == "hashing")
    #expect(value.frac == 0.3)
    #expect(value.sha256 == nil)
  }

  @Test func decodesReplies() throws {
    guard case .reply(let ok) = try ControlEvent(jsonLine: events()[12]),
          case .reply(let failed) = try ControlEvent(jsonLine: events()[13]),
          case .reply(let extras) = try ControlEvent(jsonLine: events()[14]),
          case .reply(let noID) = try ControlEvent(jsonLine: events()[16]) else {
      Issue.record("expected four reply events")
      return
    }
    #expect(ok.id == 7)
    #expect(ok.ok)
    #expect(ok.error == nil)
    #expect(ok.extras.isEmpty)
    #expect(failed.id == 8)
    #expect(!failed.ok)
    #expect(failed.error == "model a086d5249fc308bb is not downloaded")
    #expect(extras.id == 9)
    #expect(extras.extras["queued"] == .bool(true))
    #expect(extras.extras["sha256"]?.stringValue == "a086d5249fc308bb73993d1e64630c669d4c7df5bde85f42ad61902543648525")
    #expect(extras.extras["ok"] == nil)
    #expect(extras.extras["t"] == nil)
    #expect(noID.id == nil)
    #expect(!noID.ok)
  }

  @Test func decodesUnknownEvent() throws {
    guard case .unknown(let name) = try ControlEvent(jsonLine: events()[15]) else {
      Issue.record("expected an unknown event")
      return
    }
    #expect(name == "weather")
  }

  @Test func malformedLineThrows() throws {
    let line = try events()[17]
    #expect(throws: (any Error).self) { try ControlEvent(jsonLine: line) }
  }

  @Test func encodesEveryCommand() {
    let cases: [(ControlCommand, Int, String)] = [
      (.status, 1, #"{"cmd":"status","id":1}"#),
      (.catalog(refresh: true), 2, #"{"cmd":"catalog","id":2,"refresh":true}"#),
      (.catalog(refresh: false), 3, #"{"cmd":"catalog","id":3,"refresh":false}"#),
      (.download(ref: "f877d7a0ccc3cce943c76e285214c020cd65c899", sha256: nil), 4,
       #"{"cmd":"download","id":4,"ref":"f877d7a0ccc3cce943c76e285214c020cd65c899","sha256":null}"#),
      (.download(ref: nil, sha256: "a086"), 5, #"{"cmd":"download","id":5,"ref":null,"sha256":"a086"}"#),
      (.cancelDownload(sha256: "a086"), 6, #"{"cmd":"cancel_download","id":6,"sha256":"a086"}"#),
      (.importModel(path: "/Users/me/Downloads/big.onnx"), 7,
       #"{"cmd":"import","id":7,"path":"/Users/me/Downloads/big.onnx"}"#),
      (.prepare(sha256: "a086", frameSkip: 4), 8, #"{"cmd":"prepare","frame_skip":4,"id":8,"sha256":"a086"}"#),
      (.unload, 9, #"{"cmd":"unload","id":9}"#),
      (.forget(sha256: "a086", artifacts: true, model: false), 10,
       #"{"artifacts":true,"cmd":"forget","id":10,"model":false,"sha256":"a086"}"#),
      (.inventory, 11, #"{"cmd":"inventory","id":11}"#),
      (.shutdown, 12, #"{"cmd":"shutdown","id":12}"#),
    ]
    for (command, id, expected) in cases {
      let data = command.jsonLine(id: id)
      #expect(data.last == 0x0A, "\(command) must end with a newline")
      let text = String(decoding: data.dropLast(), as: UTF8.self)
      #expect(text == expected)
    }
  }
}
