#!/usr/bin/env swift
//
// Draws the Jetlink app icon and fills in Assets.xcassets/AppIcon.appiconset.
//
//   swift macos/scripts/make-icon.swift macos/build/icon.png
//
// The output is committed, so this runs once. It draws a rounded square in
// #1E6FD9 with a white cable connector glyph: two rounded rectangles (the
// plug shell and the body) joined by a line (the cable). Nothing fancy.
//
// Gotcha: `sips -z h w` takes height first, then width.

import AppKit
import CoreGraphics
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

guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
      let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                              bytesPerRow: 0, space: colorSpace,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
  FileHandle.standardError.write(Data("could not create a bitmap context\n".utf8))
  exit(1)
}

let size = CGFloat(side)
context.interpolationQuality = .high
context.setAllowsAntialiasing(true)

// macOS icons leave a margin around the rounded square.
let margin = size * 0.09
let plate = CGRect(x: margin, y: margin, width: size - 2 * margin, height: size - 2 * margin)
let plateRadius = plate.width * 0.225
let blue = CGColor(srgbRed: 0x1E / 255.0, green: 0x6F / 255.0, blue: 0xD9 / 255.0, alpha: 1.0)
context.addPath(CGPath(roundedRect: plate, cornerWidth: plateRadius, cornerHeight: plateRadius, transform: nil))
context.setFillColor(blue)
context.fillPath()

// The glyph: a plug shell on the left, a body on the right, a cable between.
context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
context.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))

let shell = CGRect(x: size * 0.215, y: size * 0.395, width: size * 0.175, height: size * 0.21)
context.addPath(CGPath(roundedRect: shell, cornerWidth: size * 0.035, cornerHeight: size * 0.035, transform: nil))
context.fillPath()

let body = CGRect(x: size * 0.575, y: size * 0.335, width: size * 0.21, height: size * 0.33)
context.addPath(CGPath(roundedRect: body, cornerWidth: size * 0.05, cornerHeight: size * 0.05, transform: nil))
context.fillPath()

context.setLineWidth(size * 0.058)
context.setLineCap(.round)
context.move(to: CGPoint(x: shell.maxX, y: size * 0.5))
context.addLine(to: CGPoint(x: body.minX, y: size * 0.5))
context.strokePath()

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
