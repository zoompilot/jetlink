import Foundation
import JetlinkKit
import JetlinkRegistry

/// The big model picked last, wherever it was picked: here (the web page, an
/// app's Use Model, `prepare` on the control channel) or on the comma, whose
/// hello says what it picked and how long ago. The newer pick wins, and the
/// hello's answer carries it, so a comma whose own pick lost takes this one
/// and both ends name the same model.
///
/// Kept beside the cache (`CacheLayout.pickURL`), so a restart remembers it.
///
/// Which is newer is settled by what the comma said last, not by clocks
/// where that is enough: a Jetson in a car is mostly offline, and at each
/// power-on its clock starts again from its last network sync, behind a pick
/// it dated in an earlier drive. So a pick of ours that the comma has named
/// (or that was its own) loses to any other the comma names now, which it
/// picked after; and while the comma still names the pick it named last
/// time (the same model, made when the comma said it was), ours was picked
/// here since and wins. Only when both ends picked since the last hello do
/// the dates decide, each by its own clock with ages crossing the link.
public final class PickStore: @unchecked Sendable {
  public struct Pick: Equatable, Sendable {
    public var sha256: String?
    public var ref: String?
    public var nbytes: Int64?
    public var name: String?
    /// When it was picked, by this machine's clock.
    public var at: TimeInterval
    /// Where: `here`, or `comma`.
    public var by: String
    /// A comma's pick: when it was made by the comma's clock, as its hello
    /// says (`at`; older commas send only the age). The same pick in two
    /// hellos has the same one, whatever this clock did in between.
    public var commaAt: TimeInterval? = nil

    /// The same model, by its catalog ref where both have one: two refs can
    /// share one export, and the comma names its pick by the ref.
    func sameModel(as other: Pick) -> Bool {
      if let ref, let theirs = other.ref { return ref == theirs }
      if let sha256, let theirs = other.sha256 { return sha256 == theirs }
      return false
    }

    /// The same pick of the same model, named by two hellos: by the comma's
    /// date where both have one, else by this clock's, which a power cut
    /// between the two can move.
    func samePick(as other: Pick) -> Bool {
      guard sameModel(as: other) else { return false }
      if let commaAt, let theirs = other.commaAt { return abs(commaAt - theirs) < 1 }
      return abs(at - other.at) < 5
    }
  }

  private let url: URL
  private let lock = NSLock()
  private var pick: Pick?
  /// The pick the comma's last hello named, nil when it named none.
  private var heard: Pick?
  private let log = ServerLog(category: "server")

  public init(layout: CacheLayout) {
    url = layout.pickURL
    if let data = try? Data(contentsOf: url), let d = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
      let at = (d["at"] as? NSNumber)?.doubleValue
    {
      pick = PickStore.parse(d, at: at, by: (d["by"] as? String) ?? "here")
      heard = (d["heard"] as? [String: Any]).flatMap { h in
        (h["at"] as? NSNumber).flatMap { PickStore.parse(h, at: $0.doubleValue, by: "comma") }
      }
    }
  }

  public func current() -> Pick? {
    lock.withLock { pick }
  }

  /// A pick made here. Says nothing when it is the model picked already.
  public func pickHere(sha256: String, ref: String?, nbytes: Int64?, name: String?, now: TimeInterval = Date().timeIntervalSince1970) {
    let made = Pick(sha256: sha256, ref: ref, nbytes: nbytes.flatMap { $0 > 0 ? $0 : nil }, name: name, at: now, by: "here")
    lock.withLock {
      if let pick, pick.by == "here", pick.sameModel(as: made) { return }
      pick = made
      save()
    }
    log.info("picked here: \(name ?? String(sha256.prefix(16)))")
  }

  /// The pick a comma's hello carries, `{ref, sha256, nbytes, name, age_s}`,
  /// against the one kept here; the newer is kept (see the type's notes), and
  /// returned for the hello's answer. A comma with no pick of its own (no
  /// `age_s`, it runs its default) does not compete.
  public func meet(_ hello: [String: Any]?, now: TimeInterval = Date().timeIntervalSince1970) -> Pick? {
    guard let hello, let age = (hello["age_s"] as? NSNumber)?.doubleValue, age.isFinite, age >= 0,
      let theirs = PickStore.parse(hello, at: now - age, by: "comma", commaAtKey: "at")
    else {
      lock.withLock {
        if heard != nil {
          heard = nil
          save()
        }
      }
      return current()
    }
    return lock.withLock {
      let before = heard
      heard = theirs
      defer { save() }
      guard let mine = pick, !mine.sameModel(as: theirs) else {
        if pick == nil { pick = theirs }
        return pick
      }
      if PickStore.commaWins(mine: mine, theirs: theirs, heardBefore: before) {
        log.info("the comma's pick is newer: \(theirs.name ?? theirs.ref ?? String(theirs.sha256?.prefix(16) ?? "?"))")
        pick = theirs
        return theirs
      }
      return mine
    }
  }

  /// Whether the comma's pick, another model than ours, was made after ours.
  static func commaWins(mine: Pick, theirs: Pick, heardBefore: Pick?) -> Bool {
    // the comma has had ours, its own or one it took from here, and names another now
    if mine.by == "comma" || heardBefore?.sameModel(as: mine) == true { return true }
    // the comma names the pick it named last time, so ours was picked here since
    if heardBefore?.samePick(as: theirs) == true { return false }
    // both picked since the last hello: only the dates can say
    return theirs.at > mine.at
  }

  /// A pick as the hello's answer carries it, with its age by this clock.
  public static func wire(_ pick: Pick?, now: TimeInterval = Date().timeIntervalSince1970) -> Any {
    guard let pick else { return NSNull() }
    return [
      "sha256": pick.sha256 ?? NSNull(), "ref": pick.ref ?? NSNull(), "nbytes": pick.nbytes.map { NSNumber(value: $0) } ?? NSNull(),
      "name": pick.name ?? NSNull(), "age_s": pythonRound(max(0, now - pick.at), 1), "by": pick.by,
    ] as [String: Any]
  }

  /// A pick from a hello or the file: a model identity at least, the rest as far as it is good.
  /// `commaAtKey` names the comma's date of it: `at` in a hello, `comma_at` in the file.
  static func parse(_ d: [String: Any], at: TimeInterval, by: String, commaAtKey: String = "comma_at") -> Pick? {
    let sha256 = (d["sha256"] as? String).flatMap { CacheLayout.isSHA256($0) ? $0 : nil }
    let ref = (d["ref"] as? String).flatMap { CacheLayout.isRef($0) ? $0 : nil }
    guard sha256 != nil || ref != nil else { return nil }
    let nbytes = (d["nbytes"] as? NSNumber).flatMap { $0.int64Value > 0 ? $0.int64Value : nil }
    let name = (d["name"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(200)) }
    let commaAt = (d[commaAtKey] as? NSNumber).map(\.doubleValue).flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
    return Pick(sha256: sha256, ref: ref, nbytes: nbytes, name: name, at: at, by: by, commaAt: commaAt)
  }

  /// Hold `lock`.
  private func save() {
    guard let pick else { return }
    var d: [String: Any] = ["at": pick.at, "by": pick.by]
    d["sha256"] = pick.sha256
    d["ref"] = pick.ref
    d["nbytes"] = pick.nbytes.map { NSNumber(value: $0) }
    d["name"] = pick.name
    d["comma_at"] = pick.commaAt
    if let heard {
      var h: [String: Any] = ["at": heard.at]
      h["ref"] = heard.ref
      h["sha256"] = heard.sha256
      h["comma_at"] = heard.commaAt
      d["heard"] = h
    }
    do {
      let data = try JSONSerialization.data(withJSONObject: d, options: [.sortedKeys])
      try data.write(to: url, options: .atomic)
    } catch {
      log.warning("could not keep the pick: \(String(describing: error))")
    }
  }
}
