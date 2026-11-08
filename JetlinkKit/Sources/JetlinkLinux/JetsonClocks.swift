#if os(Linux)
  import Foundation
  import JetlinkServer

  /// A Jetson's GPU, memory and CPU clocks, set live inside the limits of
  /// the power mode the installer named: held at its top while a comma is
  /// connected or an engine builds (`--clocks on`), and left to NVIDIA's
  /// scaling otherwise. Each clock is a lowest and a highest in sysfs (the
  /// GPU's and memory's devfreq nodes, cpufreq's policies); held is both at
  /// the top. The limits come from nvpmodel's configuration, so they hold
  /// where nvpmodel could not switch to the mode without a restart.
  public final class JetsonClocks: @unchecked Sendable {
    /// After the comma leaves, how long the clocks stay up: a handover or a
    /// cable's blip reconnects within it.
    static let dropAfter: TimeInterval = 5

    struct Node {
      let name: String
      let minFile: String
      let maxFile: String
      let clock: JetsonPower.Clock
    }

    let root: HostRoot
    let nodes: [Node]
    private let log = ServerLog(category: "clocks")
    /// Everything below runs on it, off the threads that report events.
    private let queue = DispatchQueue(label: "io.zoompilot.jetlink.clocks")
    private var connected = false
    private var building = false
    private var held: Bool?
    private var drop: DispatchWorkItem?

    /// nil off a Jetson (no GPU devfreq node): the clocks stay as they are.
    public init?(modeName: String, root: HostRoot = .system) {
      guard let gpu = gpuDevfreq(root) else { return nil }
      let limits = JetsonPower.limits(Sysfs.read(root.path("/etc/nvpmodel.conf")) ?? "", mode: modeName) ?? JetsonPower.ModeLimits()
      var found = [
        JetsonClocks.node(root, "GPU", gpu, steps: "available_frequencies", min: "min_freq", max: "max_freq", cap: limits.gpu),
        JetsonClocks.node(
          root, "memory", root.path("/sys/class/devfreq/bwmgr"), steps: "available_frequencies", min: "min_freq", max: "max_freq", cap: limits.memory),
      ]
      for policy in root.list("/sys/devices/system/cpu/cpufreq") where policy.hasPrefix("policy") {
        found.append(
          JetsonClocks.node(
            root, "CPU", root.path("/sys/devices/system/cpu/cpufreq/\(policy)"), steps: "scaling_available_frequencies", min: "scaling_min_freq",
            max: "scaling_max_freq", cap: limits.cpu))
      }
      guard let gpuNode = found[0] else { return nil }
      self.root = root
      nodes = [gpuNode] + found.dropFirst().compactMap { $0 }
      log.info("clocks inside power mode \(modeName)")
    }

    /// Off: NVIDIA's scaling, once, for clocks an earlier run held.
    public func release() {
      queue.async { self.apply(false) }
    }

    /// On: follows the comma and the builds from the host's events.
    public func follow(_ host: EngineHost) {
      release()
      host.subscribe { [self] event in
        switch event {
        case .link(let link):
          queue.async {
            self.connected = link.state == .connected; self.update()
          }
        case .engine(let engine):
          queue.async {
            self.building = engine.state == .building; self.update()
          }
        default: break
        }
      }
    }

    /// Up at once; down only once nothing has wanted them for `dropAfter`.
    private func update() {
      if connected || building {
        drop?.cancel()
        drop = nil
        apply(true)
      } else if held == true, drop == nil {
        let item = DispatchWorkItem { [self] in
          drop = nil
          if !(connected || building) { apply(false) }
        }
        drop = item
        queue.asyncAfter(deadline: .now() + JetsonClocks.dropAfter, execute: item)
      }
    }

    /// Every clock keeps the mode's top as its highest, written once; held
    /// raises its lowest to that top, free puts it back at the bottom.
    private func apply(_ hold: Bool) {
      guard hold != held else { return }
      let first = held == nil
      var failed: [String] = []
      for node in nodes {
        do throws(KernelError) {
          if first {
            // down first, so the top is never under the lowest
            try Sysfs.write(node.minFile, "\(node.clock.bottom)")
            try Sysfs.write(node.maxFile, "\(node.clock.top)")
          }
          if hold || !first { try Sysfs.write(node.minFile, "\(hold ? node.clock.top : node.clock.bottom)") }
        } catch {
          failed.append("\(node.name): \(error)")
        }
      }
      held = hold
      log.info(hold ? "clocks held at the power mode's top" : "clocks left to NVIDIA's scaling")
      if !failed.isEmpty { log.warning("cannot set every clock: \(failed.joined(separator: "; "))") }
    }

    /// `directory` is a full path; cpufreq without a table names only its ends.
    static func node(_ root: HostRoot, _ name: String, _ directory: String, steps: String, min: String, max: String, cap: Int?) -> Node? {
      var found = (Sysfs.read("\(directory)/\(steps)") ?? "").split(separator: " ").compactMap { Int($0) }
      if found.isEmpty { found = ["cpuinfo_min_freq", "cpuinfo_max_freq"].compactMap { Sysfs.readInt("\(directory)/\($0)") } }
      guard !found.isEmpty else { return nil }
      return Node(name: name, minFile: "\(directory)/\(min)", maxFile: "\(directory)/\(max)", clock: JetsonPower.Clock(steps: found, cap: cap))
    }
  }
#endif
