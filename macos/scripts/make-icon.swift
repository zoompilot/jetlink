#!/usr/bin/env swift
//
// Draws the Jetlink app icon and fills in Assets.xcassets/AppIcon.appiconset.
//
//   swift macos/scripts/make-icon.swift macos/build/icon.png
//
// The output is committed, so this runs once. The picture is a J drawn as
// openpilot's driving path, a green band between two lane lines, widest at the
// bottom and narrowing as it climbs into a USB-C plug. The greens are the ones
// openpilot's path uses (THROTTLE_COLORS in selfdrive/ui/onroad/model_renderer.py),
// and like openpilot the gradient runs up the screen, not along the path.
//
// It follows Apple's 1024 px template: an 824 px plate with continuous corners,
// centred, with its drop shadow in the margin. Coordinates below are top-left
// origin, y down, like the template.
//
// Gotcha: `sips -z h w` takes height first, then width.


import AppKit
import CoreGraphics
import CoreImage
import Foundation

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
  FileHandle.standardError.write(Data("usage: make-icon.swift OUT.png\n".utf8))
  exit(1)
}
let outputPath = arguments[1]
let side = 1024

// The appiconset sits next to this script, under Resources.
let scriptURL = URL(fileURLWithPath: arguments[0]).resolvingSymlinksInPath()
let macosDirectory = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
let iconSetDirectory = macosDirectory
  .appendingPathComponent("Resources")
  .appendingPathComponent("Assets.xcassets")
  .appendingPathComponent("AppIcon.appiconset")

guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
  FileHandle.standardError.write(Data("no sRGB colour space\n".utf8))
  exit(1)
}

let size = CGFloat(side)

// A context flipped to the template's y-down coordinates.
func makeCanvas() -> CGContext {
  guard let canvas = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                               bytesPerRow: 0, space: colorSpace,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
    FileHandle.standardError.write(Data("could not create a bitmap context\n".utf8))
    exit(1)
  }
  canvas.interpolationQuality = .high
  canvas.setAllowsAntialiasing(true)
  canvas.translateBy(x: 0, y: size)
  canvas.scaleBy(x: 1, y: -1)
  return canvas
}

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
  CGColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
          blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
}

func gradient(_ stops: [(CGFloat, CGColor)]) -> CGGradient {
  CGGradient(colorsSpace: colorSpace, colors: stops.map(\.1) as CFArray, locations: stops.map(\.0))!
}

// Fills whatever is currently clipped with a vertical gradient from y0 to y1.
func fillVertical(_ canvas: CGContext, _ stops: [(CGFloat, CGColor)], from y0: CGFloat, to y1: CGFloat) {
  canvas.drawLinearGradient(gradient(stops), start: CGPoint(x: 0, y: y0), end: CGPoint(x: 0, y: y1),
                            options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
}

// The same, left to right from x0 to x1.
func fillHorizontal(_ canvas: CGContext, _ stops: [(CGFloat, CGColor)], from x0: CGFloat, to x1: CGFloat) {
  canvas.drawLinearGradient(gradient(stops), start: CGPoint(x: x0, y: 0), end: CGPoint(x: x1, y: 0),
                            options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
}

// Draws into a fresh canvas, Gaussian-blurs it and composites the result.
// This is how the glows get the colours of the shapes that cast them.
func drawBlurred(into canvas: CGContext, sigma: CGFloat, alpha: CGFloat, _ draw: (CGContext) -> Void) {
  let layer = makeCanvas()
  draw(layer)
  let full = CGRect(x: 0, y: 0, width: size, height: size)
  guard let sharp = layer.makeImage(),
        let blur = CIFilter(name: "CIGaussianBlur", parameters: [
          kCIInputImageKey: CIImage(cgImage: sharp), kCIInputRadiusKey: sigma]),
        let output = blur.outputImage,
        let blurred = CIContext().createCGImage(output, from: full) else {
    FileHandle.standardError.write(Data("could not blur a layer\n".utf8))
    exit(1)
  }
  // The image is already upright, so draw it with the flip undone.
  canvas.saveGState()
  canvas.translateBy(x: 0, y: size)
  canvas.scaleBy(x: 1, y: -1)
  canvas.setAlpha(alpha)
  canvas.draw(blurred, in: full)
  canvas.restoreGState()
}

let context = makeCanvas()

// The plate: straight sides joined by superellipse corners that start 1.528 r
// from each corner, which is how Apple's continuous corners are built. With
// r = 185.4 this matches the system icons' mask to within anti-aliasing.
let plateInset: CGFloat = 100
let plate = CGMutablePath()
do {
  let half = (size - 2 * plateInset) / 2
  let extent = 1.528 * 185.4
  let exponent = 3.3
  let steps = 90
  var points: [CGPoint] = []
  for (index, (signX, signY)) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)].enumerated() {
    let cornerX = size / 2 + signX * half, cornerY = size / 2 + signY * half
    let centreX = cornerX - signX * extent, centreY = cornerY - signY * extent
    for step in 0...steps {
      var angle = Double.pi / 2 * Double(step) / Double(steps)
      if index % 2 == 1 { angle = Double.pi / 2 - angle }
      points.append(CGPoint(x: centreX + signX * extent * pow(cos(angle), 2 / exponent),
                            y: centreY + signY * extent * pow(sin(angle), 2 / exponent)))
    }
  }
  plate.addLines(between: points)
  plate.closeSubpath()
}


// The shadow falls down the image; shadow offsets ignore the flip, so it is
// negative here.
context.saveGState()
context.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: color(0x000000, 0.35))
context.addPath(plate)
context.setFillColor(color(0x12151B))
context.fillPath()
context.restoreGState()

context.saveGState()
context.addPath(plate)
context.clip()
fillVertical(context, [(0, color(0x1B1F27)), (1, color(0x08090C))], from: plateInset, to: size - plateInset)

// The J's centre line: a short tail, a half-circle bowl, then the stem up into
// the plug. Sampled by arc length so the band can taper evenly along it.
let stemX: CGFloat = 628, bowlRadius: CGFloat = 132, bowlY: CGFloat = 628
let tailLength: CGFloat = 10, stemTop: CGFloat = 430
let bowlX = stemX - bowlRadius
let pathLength = tailLength + .pi * bowlRadius + (bowlY - stemTop)
let samples: [(point: CGPoint, tangent: CGVector, along: CGFloat)] = (0...600).map { step in
  let s = pathLength * CGFloat(step) / 600
  if s <= tailLength {
    return (CGPoint(x: bowlX - bowlRadius, y: bowlY - tailLength + s), CGVector(dx: 0, dy: 1), s / pathLength)
  }
  if s <= tailLength + .pi * bowlRadius {
    let angle = .pi - (s - tailLength) / bowlRadius
    return (CGPoint(x: bowlX + bowlRadius * cos(angle), y: bowlY + bowlRadius * sin(angle)),
            CGVector(dx: sin(angle), dy: -cos(angle)), s / pathLength)
  }
  return (CGPoint(x: stemX, y: bowlY - (s - tailLength - .pi * bowlRadius)), CGVector(dx: 0, dy: -1), s / pathLength)
}

// A strip of the path between two offsets from the centre line, as fractions
// of the local width. The width falls from 140 at the tail to 96 at the plug,
// the way the path narrows as it recedes up the screen.
func pathStrip(_ from: CGFloat, _ to: CGFloat) -> CGPath {
  func edge(_ offset: CGFloat) -> [CGPoint] {
    samples.map { sample in
      let width = 140 + (96 - 140) * sample.along
      return CGPoint(x: sample.point.x - sample.tangent.dy * width * offset,
                     y: sample.point.y + sample.tangent.dx * width * offset)
    }
  }
  let strip = CGMutablePath()
  strip.addLines(between: edge(from) + edge(to).reversed())
  strip.closeSubpath()
  return strip
}
let laneWidth: CGFloat = 0.13, laneGap: CGFloat = 0.06
let band = pathStrip(-0.5 + laneWidth + laneGap, 0.5 - laneWidth - laneGap)
let lanes = [pathStrip(-0.5, -0.5 + laneWidth), pathStrip(0.5 - laneWidth, 0.5)]

let pathGreen = 0x0DF87A as UInt32, pathLime = 0x72FF5C as UInt32
drawBlurred(into: context, sigma: 26, alpha: 0.3) { canvas in
  canvas.setFillColor(color(pathGreen))
  for strip in [band] + lanes {
    canvas.addPath(strip)
    canvas.fillPath()
  }
}
func fill(_ strip: CGPath, _ stops: [(CGFloat, CGColor)]) {
  context.saveGState()
  context.addPath(strip)
  context.clip()
  fillVertical(context, stops, from: 900, to: 420)
  context.restoreGState()
}
fill(band, [(0, color(pathGreen, 0.62)), (1, color(pathLime, 0.52))])
for lane in lanes {
  fill(lane, [(0, color(pathGreen)), (1, color(pathLime, 0.95))])
}

// The USB-C plug, pointing up and tipped slightly toward the viewer so the
// end of the shell shows as the connector's pill-shaped opening.
let shellWidth: CGFloat = 112, shellLength: CGFloat = 84, shellTop: CGFloat = 222, openingHeight: CGFloat = 40
let bootWidth: CGFloat = 150, bootLength: CGFloat = 150
let shellLeft = stemX - shellWidth / 2
let bootTop = shellTop + shellLength
let bootLeft = stemX - bootWidth / 2, bootRight = stemX + bootWidth / 2, bootBottom = bootTop + bootLength

context.saveGState()
context.clip(to: CGRect(x: shellLeft, y: shellTop, width: shellWidth, height: shellLength + 20))
fillHorizontal(context, [(0, color(0x7E8896)), (0.18, color(0xC9D0D9)), (0.42, color(0xF7F9FB)),
                         (0.8, color(0xA9B2BE)), (1, color(0x6D7683))], from: shellLeft, to: shellLeft + shellWidth)
context.restoreGState()
context.setFillColor(color(0x000000, 0.35))
context.fill(CGRect(x: shellLeft, y: bootTop - 2, width: shellWidth, height: 6))

// The overmold narrows into the cable at the bottom.
let boot = CGMutablePath()
do {
  let neck = bootWidth * 0.72 / 2
  boot.move(to: CGPoint(x: bootLeft, y: bootTop + 36))
  boot.addQuadCurve(to: CGPoint(x: bootLeft + 36, y: bootTop), control: CGPoint(x: bootLeft, y: bootTop))
  boot.addLine(to: CGPoint(x: bootRight - 36, y: bootTop))
  boot.addQuadCurve(to: CGPoint(x: bootRight, y: bootTop + 36), control: CGPoint(x: bootRight, y: bootTop))
  boot.addLine(to: CGPoint(x: bootRight, y: bootTop + bootLength * 0.62))
  boot.addCurve(to: CGPoint(x: stemX + neck, y: bootBottom - 18),
                control1: CGPoint(x: bootRight, y: bootTop + bootLength * 0.86),
                control2: CGPoint(x: stemX + neck, y: bootTop + bootLength * 0.8))
  boot.addQuadCurve(to: CGPoint(x: stemX + neck - 18, y: bootBottom), control: CGPoint(x: stemX + neck, y: bootBottom))
  boot.addLine(to: CGPoint(x: stemX - neck + 18, y: bootBottom))
  boot.addQuadCurve(to: CGPoint(x: stemX - neck, y: bootBottom - 18), control: CGPoint(x: stemX - neck, y: bootBottom))
  boot.addCurve(to: CGPoint(x: bootLeft, y: bootTop + bootLength * 0.62),
                control1: CGPoint(x: stemX - neck, y: bootTop + bootLength * 0.8),
                control2: CGPoint(x: bootLeft, y: bootTop + bootLength * 0.86))
  boot.closeSubpath()
}
context.saveGState()
context.addPath(boot)
context.clip()
fillHorizontal(context, [(0, color(0x23272F)), (0.35, color(0x3E4452)), (1, color(0x15181D))],
               from: bootLeft, to: bootRight)
context.restoreGState()
context.addPath(boot)
context.setStrokeColor(color(0xFFFFFF, 0.16))
context.setLineWidth(3)
context.strokePath()

// The opening: a bright rim, the dark inside, and the tongue across it.
let rimRadius = openingHeight / 2
let opening = CGRect(x: shellLeft, y: shellTop - rimRadius, width: shellWidth, height: openingHeight)
context.saveGState()
context.addPath(CGPath(roundedRect: opening, cornerWidth: rimRadius, cornerHeight: rimRadius, transform: nil))
context.clip()
fillVertical(context, [(0, color(0xFFFFFF)), (1, color(0xBCC4CE))], from: opening.minY, to: opening.maxY)
context.restoreGState()
let inside = opening.insetBy(dx: 9, dy: 4.95)
context.addPath(CGPath(roundedRect: inside, cornerWidth: rimRadius - 4.95, cornerHeight: rimRadius - 4.95, transform: nil))
context.setFillColor(color(0x07090D))
context.fillPath()
let tongue = CGRect(x: shellLeft + 30, y: shellTop - 5, width: shellWidth - 60, height: 10)
context.addPath(CGPath(roundedRect: tongue, cornerWidth: 5, cornerHeight: 5, transform: nil))
context.setFillColor(color(0x4A5366))
context.fillPath()
context.restoreGState()

// A thin rim, lit from above, keeps the dark plate distinct on a dark Dock.
context.saveGState()
context.addPath(plate)
context.setLineWidth(3)
context.replacePathWithStrokedPath()
context.clip()
fillVertical(context, [(0, color(0xFFFFFF, 0.28)), (0.5, color(0xFFFFFF, 0.04)), (1, color(0xFFFFFF, 0.10))],
             from: plateInset, to: size - plateInset)
context.restoreGState()


guard let image = context.makeImage() else {
  FileHandle.standardError.write(Data("could not render the icon\n".utf8))
  exit(1)
}
let representation = NSBitmapImageRep(cgImage: image)
representation.size = NSSize(width: side, height: side)
guard let png = representation.representation(using: .png, properties: [:]) else {
  FileHandle.standardError.write(Data("could not encode the icon as PNG\n".utf8))
  exit(1)
}
let outputURL = URL(fileURLWithPath: outputPath)
try? FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
try png.write(to: outputURL)
print("wrote \(outputPath)")

// The ten standard macOS entries: 16, 32, 128, 256, 512 at 1x and 2x.
struct Entry {
  let idiom = "mac"
  let size: Int
  let scale: Int
  var pixels: Int { size * scale }
  var filename: String { "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png" }
}
let entries = [16, 32, 128, 256, 512].flatMap { [Entry(size: $0, scale: 1), Entry(size: $0, scale: 2)] }

try FileManager.default.createDirectory(at: iconSetDirectory, withIntermediateDirectories: true)
for entry in entries {
  let destination = iconSetDirectory.appendingPathComponent(entry.filename)
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/usr/bin/sips")
  process.arguments = ["-z", String(entry.pixels), String(entry.pixels), outputPath, "--out", destination.path]
  process.standardOutput = FileHandle.nullDevice
  process.standardError = FileHandle.nullDevice
  try process.run()
  process.waitUntilExit()
  guard process.terminationStatus == 0 else {
    FileHandle.standardError.write(Data("sips failed for \(entry.filename)\n".utf8))
    exit(1)
  }
}

var images: [String] = []
for entry in entries {
  images.append("""
        {
          "filename" : "\(entry.filename)",
          "idiom" : "mac",
          "scale" : "\(entry.scale)x",
          "size" : "\(entry.size)x\(entry.size)"
        }
    """)
}
let contents = """
  {
    "images" : [
  \(images.joined(separator: ",\n"))
    ],
    "info" : {
      "author" : "xcode",
      "version" : 1
    }
  }

  """
try contents.write(to: iconSetDirectory.appendingPathComponent("Contents.json"), atomically: true, encoding: .utf8)
print("wrote \(entries.count) sizes into \(iconSetDirectory.path)")
