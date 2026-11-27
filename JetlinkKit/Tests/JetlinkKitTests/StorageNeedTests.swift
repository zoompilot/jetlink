import Foundation
import Testing

@testable import JetlinkKit

/// What a model needs on disk where it is prepared for CoreML, from its size.
struct StorageNeedTests {
  static let gb: Int64 = 1_000_000_000

  /// Light Speed 2 is 768,749,892 bytes: the shares measured on an iPad Air (M1).
  @Test func lightSpeed2() {
    let need = StorageNeed.coreML(modelBytes: 768_749_892)
    #expect(need.prepared == 2 * 768_749_892 && need.buildScratch == 768_749_892 && need.neuralEngine == 768_749_892)
    #expect(need.headroom == StorageNeed.headroomBytes)
    #expect(need.bytes(for: .download) == 5 * 768_749_892 + StorageNeed.headroomBytes)
    #expect(need.bytes(for: .prepare) == 4 * 768_749_892 + StorageNeed.headroomBytes)
    #expect(need.bytes(for: .run) == 768_749_892 + StorageNeed.headroomBytes)
  }

  /// Lebowski, 1.76 GB, needs about 11.8 GB free to get, and 4.8 GB to keep.
  @Test func lebowski() {
    let need = StorageNeed.coreML(modelBytes: 1_757_355_221)
    #expect(StorageNeed.gigabytes(need.bytes(for: .download)) == "11.8 GB")
    #expect(StorageNeed.gigabytes(need.bytes(for: .run)) == "4.8 GB")
  }

  /// Compare Processors prepares in turn: a copy per processor, a Neural
  /// Engine copy for those on it, and one build's working copy at a time.
  @Test func comparingProcessors() {
    let need = StorageNeed.coreML(modelBytes: Self.gb)
    #expect(need.toPrepare(profiles: 0, onNeuralEngine: 0) == (0, 0))
    let three = need.toPrepare(profiles: 3, onNeuralEngine: 2)
    #expect(three.need == 3 * 2 * Self.gb + 2 * Self.gb + Self.gb + StorageNeed.headroomBytes)
    #expect(three.writes == 3 * 2 * Self.gb + Self.gb)
  }

  @Test func enoughRoomIsNoShortfall() {
    let need = StorageNeed.coreML(modelBytes: Self.gb)
    #expect(need.shortfall(for: .download, free: need.bytes(for: .download)) == nil)
    #expect(need.shortfall(for: .run, free: 100 * Self.gb) == nil)
  }

  @Test func aShortfallSaysWhatItNeedsAndWhatIsFree() throws {
    let need = StorageNeed.coreML(modelBytes: Self.gb)
    let download = try #require(need.shortfall(for: .download, free: 2 * Self.gb, name: "Lebowski"))
    #expect(download.step == .download && download.need == 8 * Self.gb && download.free == 2 * Self.gb)
    #expect(download.message.hasPrefix("Lebowski needs about 8.0 GB free to download and prepare"))
    #expect(download.message.contains("2.0 GB is free") && download.message.hasSuffix("Free some space and try again."))
    let prepare = try #require(need.shortfall(for: .prepare, free: Self.gb))
    #expect(prepare.message.hasPrefix("This model needs about 7.0 GB free to prepare"))
    let run = try #require(need.shortfall(for: .run, free: Self.gb))
    #expect(run.message.contains("to keep running on the Neural Engine") && run.message.contains("runs on the CPU"))
  }

  /// Only a step whose own files would not fit is refused: the free figure
  /// is the system's guess, so the rest is a warning to go past.
  @Test func onlyWhatCannotFitBlocks() throws {
    let need = StorageNeed.coreML(modelBytes: Self.gb)
    #expect(need.writes(for: .download) == 4 * Self.gb && need.writes(for: .prepare) == 3 * Self.gb && need.writes(for: .run) == 0)
    #expect(try #require(need.shortfall(for: .download, free: 4 * Self.gb - 1)).blocking)
    let warned = try #require(need.shortfall(for: .download, free: 4 * Self.gb))
    #expect(!warned.blocking && warned.message.contains("It fits, but"))
    #expect(try #require(need.shortfall(for: .prepare, free: 3 * Self.gb - 1)).blocking)
    #expect(try #require(need.shortfall(for: .prepare, free: 3 * Self.gb)).blocking == false)
    #expect(try #require(need.shortfall(for: .run, free: 0)).blocking == false)
  }

  /// Frames near the budget say nothing; the CPU's 400 ms say what is wrong,
  /// with the storage figures when they are short.
  @Test func slowFramesSayWhy() throws {
    #expect(SlowFrames.warning(meanMs: 38, frames: 600, storage: nil) == nil)
    #expect(SlowFrames.warning(meanMs: 95, frames: 600, storage: nil) == nil)
    #expect(SlowFrames.warning(meanMs: 410, frames: 0, storage: nil) == nil)
    let plain = try #require(SlowFrames.warning(meanMs: 410, frames: 600, storage: nil))
    #expect(plain.hasPrefix("Frames take 410 ms, 8 times the 50 ms budget"))
    #expect(plain.contains("for want of free space"))
    let short = StorageNeed.coreML(modelBytes: 768_749_892).shortfall(for: .run, free: 500_000_000, name: "Light Speed 2")
    let withStorage = try #require(SlowFrames.warning(meanMs: 410, frames: 600, storage: short))
    #expect(withStorage.contains("Light Speed 2 needs about 3.8 GB free") && withStorage.contains("0.5 GB is free"))
  }

  /// A server from before the Neural Engine's copies were counted sends no such key.
  @Test func anOlderServersDiskDecodes() throws {
    let json = Data(#"{"modelsBytes":1,"enginesBytes":2,"freeBytes":3}"#.utf8)
    let disk = try JSONDecoder().decode(InventoryDisk.self, from: json)
    #expect(disk == InventoryDisk(modelsBytes: 1, enginesBytes: 2, freeBytes: 3))
    let counted = InventoryDisk(modelsBytes: 1, enginesBytes: 2, freeBytes: 3, neuralEngineBytes: 4)
    let round = try JSONDecoder().decode(InventoryDisk.self, from: JSONEncoder().encode(counted))
    #expect(round.neuralEngineBytes == 4)
    // none to count, none sent: the payload other servers send
    let none = try JSONSerialization.jsonObject(with: JSONEncoder().encode(InventoryDisk(modelsBytes: 1, enginesBytes: 2, freeBytes: 3))) as? [String: Any]
    #expect(none?.keys.sorted() == ["enginesBytes", "freeBytes", "modelsBytes"])
  }
}
