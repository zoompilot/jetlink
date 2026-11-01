import Foundation
import JetlinkRegistry
import JetlinkTestSupport
import Testing

@testable import JetlinkServer

@Suite("Picks on both ends")
struct PickStoreTests {
  static let refA = String(repeating: "a", count: 40)
  static let refB = String(repeating: "b", count: 40)
  static let shaA = String(repeating: "1", count: 64)
  static let shaB = String(repeating: "2", count: 64)

  func comma(_ ref: String?, sha256: String? = nil, ago: Double?, at: Double? = nil) -> [String: Any] {
    var d: [String: Any] = ["name": "Comma's"]
    d["ref"] = ref
    d["sha256"] = sha256
    d["nbytes"] = 766_000_000
    d["age_s"] = ago
    d["at"] = at
    return d
  }

  @Test("With no pick here the comma's is taken, and kept over a restart")
  func takesTheComma() throws {
    let dir = try TemporaryDirectory()
    let store = PickStore(layout: CacheLayout(root: dir.url))
    let now = 1_000_000.0
    let pick = try #require(store.meet(comma(Self.refA, sha256: Self.shaA, ago: 30), now: now))
    #expect(pick.ref == Self.refA && pick.sha256 == Self.shaA && pick.by == "comma" && pick.at == now - 30)
    #expect(PickStore(layout: CacheLayout(root: dir.url)).current() == pick)
  }

  @Test("The newer pick wins, whichever end made it")
  func newerWins() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    let now = 1_000_000.0
    store.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: 5, name: "Beta", now: now - 100)
    // an older pick on the comma loses, and the answer names this end's
    #expect(store.meet(comma(Self.refA, ago: 500), now: now)?.ref == Self.refB)
    // a newer one wins
    #expect(store.meet(comma(Self.refA, ago: 10), now: now)?.ref == Self.refA)
    #expect(store.current()?.by == "comma")
    // and a pick here after it wins again
    store.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: 5, name: "Beta", now: now + 1)
    #expect(store.meet(comma(Self.refA, ago: 10), now: now + 2)?.ref == Self.refB)
  }

  @Test("A pick the comma has had loses to the next it names, however this clock moved")
  func clockStartsAgainBehind() throws {
    let dir = try TemporaryDirectory()
    // a drive: the comma's Lightspeed, met at 10 000 by this clock
    #expect(PickStore(layout: CacheLayout(root: dir.url)).meet(comma(Self.refB, ago: 5, at: 1_700_000_000), now: 10_000)?.ref == Self.refB)
    // a power cut, and the clock starts again behind it; the comma picked Alpha a minute ago
    let store = PickStore(layout: CacheLayout(root: dir.url))
    #expect(store.meet(comma(Self.refA, ago: 60, at: 1_700_090_000), now: 9_000)?.ref == Self.refA)
    // and the same goes for a pick made here once the comma has named it
    store.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: 5, name: "Beta", now: 9_100)
    #expect(store.meet(comma(Self.refB, ago: 1, at: 1_700_090_200), now: 9_200)?.by == "here")
    let later = PickStore(layout: CacheLayout(root: dir.url))
    #expect(later.meet(comma(Self.refA, ago: 30, at: 1_700_190_000), now: 5_000)?.ref == Self.refA)
  }

  @Test("While the comma names the pick it named last, one made here since wins, however this clock moved")
  func pickedHereSince() throws {
    let dir = try TemporaryDirectory()
    _ = PickStore(layout: CacheLayout(root: dir.url)).meet(comma(Self.refA, ago: 600, at: 1_700_000_000), now: 10_000)
    let store = PickStore(layout: CacheLayout(root: dir.url))
    store.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: 5, name: "Beta", now: 10_100)
    // a power cut before the comma's next hello; its pick is the same one, aged
    let later = PickStore(layout: CacheLayout(root: dir.url))
    #expect(later.meet(comma(Self.refA, ago: 90_000, at: 1_700_000_000), now: 2_000)?.ref == Self.refB)
    // a comma too old to date its pick: the same pick by this clock, within one drive
    let old = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    _ = old.meet(comma(Self.refA, ago: 600), now: 10_000)
    old.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: 5, name: "Beta", now: 10_100)
    #expect(old.meet(comma(Self.refA, ago: 800.04), now: 10_200)?.ref == Self.refB)
  }

  @Test("The same model picked again on the comma is a new pick")
  func pickedAgainOnTheComma() throws {
    // a model the comma does not list drives without being taken into its slot,
    // so the comma names its old pick until the user picks it again
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    _ = store.meet(comma(Self.refA, ago: 600, at: 1_700_000_000), now: 10_000)
    store.pickHere(sha256: Self.shaB, ref: nil, nbytes: 5, name: "My export", now: 10_100)
    #expect(store.meet(comma(Self.refA, ago: 700, at: 1_700_000_000), now: 10_100)?.sha256 == Self.shaB)
    #expect(store.meet(comma(Self.refA, ago: 5, at: 1_700_000_300), now: 10_300)?.ref == Self.refA)
  }

  @Test("A comma with no pick of its own does not compete")
  func noPickOfItsOwn() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    #expect(store.meet(comma(Self.refA, ago: nil)) == nil)
    #expect(store.meet(nil) == nil)
    store.pickHere(sha256: Self.shaB, ref: Self.refB, nbytes: nil, name: nil)
    #expect(store.meet(comma(Self.refA, ago: nil))?.ref == Self.refB)
  }

  @Test("The same model from the comma, however recent, is no change")
  func sameModel() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    let now = 1_000_000.0
    store.pickHere(sha256: Self.shaA, ref: nil, nbytes: nil, name: nil, now: now - 100)
    let pick = try #require(store.meet(comma(nil, sha256: Self.shaA, ago: 1), now: now))
    #expect(pick.by == "here" && pick.at == now - 100)
  }

  @Test("Two refs on one export are two picks")
  func sharedExport() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    let now = 1_000_000.0
    store.pickHere(sha256: Self.shaA, ref: Self.refB, nbytes: 5, name: "Flip Flop", now: now - 100)
    #expect(store.meet(comma(Self.refA, sha256: Self.shaA, ago: 10), now: now)?.ref == Self.refA)
  }

  @Test("Nothing that is not a model identity is taken")
  func refusesJunk() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    #expect(store.meet(comma("../etc", sha256: "x", ago: 1)) == nil)
    #expect(store.meet(comma(Self.refA, ago: -5)) == nil)
    #expect(store.meet(["ref": Self.refA, "age_s": "soon"]) == nil)
  }

  @Test("The hello carries the pick with its age by this clock")
  func wire() throws {
    let store = PickStore(layout: CacheLayout(root: try TemporaryDirectory().url))
    store.pickHere(sha256: Self.shaA, ref: Self.refA, nbytes: 766, name: "Alpha", now: 1000)
    let d = try #require(PickStore.wire(store.current(), now: 1060.04) as? [String: Any])
    #expect(d["ref"] as? String == Self.refA && d["sha256"] as? String == Self.shaA && d["by"] as? String == "here")
    #expect((d["nbytes"] as? NSNumber)?.int64Value == 766 && (d["age_s"] as? Double) == 60)
    #expect(PickStore.wire(nil) is NSNull)
  }
}

@Suite("A way to the internet")
struct ReachabilityTests {
  static let header = "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"

  @Test("A default route counts, on anything but loopback")
  func defaultRoute() {
    let wifi = Self.header + "wlan0\t00000000\t0101A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n"
    let lan = Self.header + "wlan0\t0001A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n"
    let lo = Self.header + "lo\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0\n"
    #expect(Reachability.defaultRoute(ipv4: wifi, ipv6: ""))
    #expect(!Reachability.defaultRoute(ipv4: lan, ipv6: ""))
    #expect(!Reachability.defaultRoute(ipv4: lo, ipv6: ""))
    let zero = String(repeating: "0", count: 32)
    let v6 = "\(zero) 00 \(zero) 00 fe800000000000000000000000000001 00000400 00000001 00000000 00000003 wlan0\n"
    let unreachable = "\(zero) 00 \(zero) 00 \(zero) ffffffff 00000001 00000000 00200200 lo\n"
    #expect(Reachability.defaultRoute(ipv4: lan, ipv6: v6))
    #expect(!Reachability.defaultRoute(ipv4: lan, ipv6: unreachable))
  }
}
