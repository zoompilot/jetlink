import AppKit

/// The app icon's mark for the menu bar: the J drawn as openpilot's path, rising
/// into a USB-C plug, as a template image the menu bar tints for itself. It is
/// filled while a comma is connected, drawn in outline while Jetlink waits for
/// one, and faded when the server is not running.
///
/// The geometry is the icon's (macos/scripts/make-icon.swift, in its 1024 px
/// units), with the band a little wider so its outline survives at 18 points.
enum MenuBarGlyph {
  enum State {
    case off, waiting, connected
  }

  static func image(_ state: State) -> NSImage {
    switch state {
    case .off: off
    case .waiting: waiting
    case .connected: connected
    }
  }

  private static let off = make(.off)
  private static let waiting = make(.waiting)
  private static let connected = make(.connected)

  static let size = NSSize(width: 13, height: 18)

  private static func make(_ state: State) -> NSImage {
    let image = NSImage(size: size, flipped: true) { _ in
      guard let context = NSGraphicsContext.current?.cgContext else { return false }
      draw(state, in: context)
      return true
    }
    image.isTemplate = true
    image.accessibilityDescription = "Jetlink"
    return image
  }

  // The mark's extent in icon units: the tail's outer edge to the plug's
  // right side, and the plug's opening to the bottom of the bowl.
  private static let bounds = CGRect(x: 284, y: 202, width: 436, height: 633)
  /// Points per icon unit, so the mark is 16.5 points tall.
  private static let scale: CGFloat = 16.5 / 633
  /// The outline's line width, in points: about an SF Symbol's regular stroke.
  private static let outline: CGFloat = 1.15

  private static func draw(_ state: State, in context: CGContext) {
    context.saveGState()
    defer { context.restoreGState() }
    context.translateBy(
      x: (size.width - bounds.width * scale) / 2,
      y: (size.height - bounds.height * scale) / 2)
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: -bounds.minX, y: -bounds.minY)

    if state == .off { context.setAlpha(0.45) }
    context.beginTransparencyLayer(auxiliaryInfo: nil)
    defer { context.endTransparencyLayer() }

    context.setFillColor(.black)
    context.addPath(strip(inset: 0))
    context.fillPath()
    if state != .connected {
      context.setBlendMode(.clear)
      context.addPath(strip(inset: outline / scale))
      context.fillPath()
      context.setBlendMode(.normal)
    }

    // The plug covers the top of the stem: a narrow tip on a boot clearly
    // wider than the cable, which is what makes it read as a plug and not as
    // the dot of a j at this size.
    let shell = CGRect(x: 570, y: 202, width: 116, height: 112)
    context.addPath(CGPath(roundedRect: shell, cornerWidth: 22, cornerHeight: 22, transform: nil))
    context.fillPath()
    let boot = CGRect(x: 536, y: 300, width: 184, height: 152)
    context.addPath(CGPath(roundedRect: boot, cornerWidth: 40, cornerHeight: 40, transform: nil))
    context.fillPath()
  }

  /// The band between the lane lines, shrunk by `inset` icon units on each side.
  private static func strip(inset: CGFloat) -> CGPath {
    let stemX: CGFloat = 628, bowlRadius: CGFloat = 132, bowlY: CGFloat = 628
    let tailLength: CGFloat = 10, stemTop: CGFloat = 430
    let bowlX = stemX - bowlRadius
    let length = tailLength + .pi * bowlRadius + (bowlY - stemTop)
    var left: [CGPoint] = []
    var right: [CGPoint] = []
    for step in 0...120 {
      let s = length * CGFloat(step) / 120
      let point: CGPoint
      let tangent: CGVector
      if s <= tailLength {
        point = CGPoint(x: bowlX - bowlRadius, y: bowlY - tailLength + s)
        tangent = CGVector(dx: 0, dy: 1)
      } else if s <= tailLength + .pi * bowlRadius {
        let angle = .pi - (s - tailLength) / bowlRadius
        point = CGPoint(x: bowlX + bowlRadius * cos(angle), y: bowlY + bowlRadius * sin(angle))
        tangent = CGVector(dx: sin(angle), dy: -cos(angle))
      } else {
        point = CGPoint(x: stemX, y: bowlY - (s - tailLength - .pi * bowlRadius))
        tangent = CGVector(dx: 0, dy: -1)
      }
      // Wider than the icon's 140 to 96, which would leave no room inside
      // an outline this small.
      let half = (160 + (130 - 160) * s / length) / 2 - inset
      left.append(CGPoint(x: point.x - tangent.dy * half, y: point.y + tangent.dx * half))
      right.append(CGPoint(x: point.x + tangent.dy * half, y: point.y - tangent.dx * half))
    }
    let path = CGMutablePath()
    path.addLines(between: left + right.reversed())
    path.closeSubpath()
    return path
  }
}
