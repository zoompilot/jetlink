#!/usr/bin/env swift
//
// Draws the iPhone app icon into Resources/Assets.xcassets/AppIcon.appiconset.
//
//   swift ios/scripts/make-icon.swift
//
// The Mac icon's glyph and blue (macos/scripts/make-icon.swift), drawn
// full-bleed and opaque: iOS masks the corners itself and rejects an icon
// with an alpha channel. The output is committed, so this runs once.

import AppKit
import CoreGraphics
import Foundation

let side = 1024
let scriptURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let iosDirectory = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
let output = iosDirectory.appendingPathComponent("Resources/Assets.xcassets/AppIcon.appiconset/icon_1024.png")

guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
  let context = CGContext(
    data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0, space: colorSpace,
    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
else {
  FileHandle.standardError.write(Data("could not create a bitmap context\n".utf8))
  exit(1)
}

let size = CGFloat(side)
context.interpolationQuality = .high
context.setAllowsAntialiasing(true)
context.setFillColor(CGColor(srgbRed: 0x1E / 255.0, green: 0x6F / 255.0, blue: 0xD9 / 255.0, alpha: 1.0))
context.fill(CGRect(x: 0, y: 0, width: size, height: size))

// The glyph at the Mac icon's proportions: a plug shell, a body, a cable.
let white = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)
context.setFillColor(white)
context.setStrokeColor(white)
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

guard let image = context.makeImage(),
  let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
else {
  FileHandle.standardError.write(Data("could not render the icon\n".utf8))
  exit(1)
}
try png.write(to: output)
print("wrote \(output.path)")
