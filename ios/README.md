# Jetlink for iPhone development

To install and use the app, follow the [iPhone user guide](../docs/iphone-app.md).
This page covers building and testing it from source. Run the commands from the
repository root.

## Layout

The app is a thin shell over the `JetlinkKit` Swift package, which every
platform shares:

| Module | What it is |
| --- | --- |
| `JetlinkKit` | The control protocol's types, the model list and its store |
| `JetlinkUI` | SwiftUI both apps draw with: the frame budget, the headroom ring, badges |
| `JetlinkONNX` | Reading and preparing a model's ONNX for CoreML, without the onnx package |
| `JetlinkRegistry` | sunnypilot's model catalog, LFS downloads, imports, the cache layout |
| `JetlinkServer` | The server: wire protocol, TCP, session, queues, control |
| `JetlinkORT` | onnxruntime and its profiles (`OrtBackend`, `OrtProfile`): CoreML here |

`ios/Jetlink` holds only what is the phone's own: the dashboard, the Models and
Settings screens, the server's lifecycle in the app, and the phone's health.

## Build

```
brew install xcodegen
cp ios/Config/Local.xcconfig.example ios/Config/Local.xcconfig   # your team and bundle id
make -C ios open
```

Signing comes from `ios/Config/Signing.xcconfig`, and your own team and bundle
identifier from `Local.xcconfig` beside it, which git ignores. Leave Signing &
Capabilities alone: it would write them into the committed project. The
**Jetlink** scheme runs Release, since frame times from an unoptimized build
mean little; **Jetlink Debug** runs Debug for the debugger.

Xcode builds for iPhone only once the iOS platform is installed (Xcode >
Settings > Components). `make -C ios build` compiles for a device without
signing, and needs the same.

Without the platform, `make -C ios typecheck` compiles and links the app's
sources for arm64 iOS through SwiftPM, against the SDK Xcode ships with.

## Test

The server is tested on the Mac, where it is the same code:

```
cd JetlinkKit && swift test
```

The suites check the wire bytes against `jetlink/protocol.py` and run the whole
server on onnxruntime's CPU provider against golden frames
([conformance](../docs/conformance.md)).

To run the Swift server with CoreML against `scripts/bench_link.py` or
`verify_parity.py`, use `jetlink-server`, which listens on TCP port 5599:

```
swift build -c release --package-path JetlinkKit --product jetlink-server
JetlinkKit/.build/release/jetlink-server --cache /tmp/jetlink-cache
python3 scripts/bench_link.py --host 127.0.0.1 --onnx big_driving_supercombo.onnx --rate 20
```

The app listens the same way only with **Connect over Wi-Fi** on, in its
developer settings (seven taps on **Version** in Settings); the address and
port are listed under it. A comma with Jetlink set to Wi-Fi dials it there too.

## TestFlight

Uploading a build to App Store Connect: [publishing](../docs/publishing.md#iphone-app-on-testflight).

## Install on a device from source

Needs a Mac with Xcode 26 and the iOS 26 platform. A free Apple account is
enough.

1. In Xcode, open **Settings > Components** and install the **iOS 26**
   platform if it is missing.
2. Under **Settings > Accounts**, add your Apple ID. A free account shows as a
   **Personal Team**, with the ten-character team ID beside it.
3. In a checkout, copy `ios/Config/Local.xcconfig.example` to
   `ios/Config/Local.xcconfig` and set your team ID and your own bundle
   identifier. Git ignores the file. Do not set them under Signing &
   Capabilities (that writes them into the project file).
4. On the iPhone, turn on **Settings > Privacy & Security > Developer Mode** and
   restart. The switch appears once the phone has been plugged into a Mac with
   Xcode open.
5. Install xcodegen (`brew install xcodegen`) and run `make -C ios open`.
   Connect the iPhone, select it as the run destination and click **Run**. The
   first build fetches onnxruntime.
6. Trust the developer before the first launch: on the iPhone,
   **Settings > General > VPN & Device Management**, your Apple ID, **Trust**.

A free account's install stops opening after **7 days**. Click **Run** again to
renew it; models and settings are kept. A paid membership's install lasts a year.
