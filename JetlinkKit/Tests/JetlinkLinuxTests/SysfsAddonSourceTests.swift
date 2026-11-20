#if os(Linux)
  import Foundation
  import JetlinkKit
  import Testing

  @testable import JetlinkLinux
  @testable import JetlinkServer

  /// An add-on's interface beside Jetlink's own on the comma's gadget.
  @Suite("Add-on interfaces", .serialized)
  struct SysfsAddonSourceTests {
    typealias D = Descriptors
    static let vasm: [UInt8] = [0xff, 0x56, 0x41]
    static let composite =
      D.device
      + D.configuration(1, D.interface(0, D.endpoint(0x81), D.endpoint(0x01)), D.interface(1, kind: vasm, D.endpoint(0x82), D.endpoint(0x02)))

    @Test("pick(kind:) takes that class's interface and no other")
    func pick() throws {
      #expect(try GadgetDescriptors.pick(Self.composite, kind: Self.vasm) == GadgetDescriptors(interface: 1, inEndpoint: 0x82, outEndpoint: 0x02))
      #expect(try GadgetDescriptors.pick(Self.composite) == GadgetDescriptors(interface: 0, inEndpoint: 0x81, outEndpoint: 0x01))
      #expect(throws: LinkError.self) { try GadgetDescriptors.pick(D.comma, kind: Self.vasm) }
    }

    @Test("An add-on claims only its own interface, on a descriptor of its own, and lets it go alone")
    func claimsBeside() throws {
      let bus = Bus()
      bus.plug(device: 3, descriptors: Self.composite)
      #expect(bus.gadget.present())
      _ = try bus.gadget.open()
      let addon = SysfsAddonSource(kind: Self.vasm, root: bus.tree.root, node: bus.node)
      #expect(addon.present())
      _ = try addon.open()
      _ = try addon.open()
      #expect(bus.node.calls.value == ["open 002/003", "claim 0", "open 002/003", "claim 1"])
      #expect(bus.node.descriptorsOpen.value.count == 2)
      bus.unplug(device: 3)
      #expect(!addon.present())
      #expect(bus.node.calls.value.suffix(2) == ["release 1", "close"])
      #expect(!bus.node.calls.value.contains("release 0"))
    }

    @Test("A gadget without the add-on's interface: nothing claimed, and the descriptor closed")
    func missing() {
      let bus = Bus()
      bus.plug(device: 3)
      let addon = SysfsAddonSource(kind: Self.vasm, root: bus.tree.root, node: bus.node)
      #expect(addon.present())
      #expect(throws: LinkError.self) { try addon.open() }
      #expect(bus.node.calls.value == ["open 002/003", "close"])
    }
  }
#endif
