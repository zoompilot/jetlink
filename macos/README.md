# Jetlink for Mac

The macOS app that runs the Jetlink inference server, shows what the comma
sees, and manages the large driving models.

## Build

```
brew install xcodegen libusb
make -C macos app
```

`make app` generates the Xcode project, builds `macos/build/python` (an
embedded CPython 3.14.7 with the pinned wheels, tinygrad from git, the jetlink
package and libusb), builds Release, and signs everything ad hoc. The result is
`macos/build/Jetlink.app`. The first run downloads about 200 MB into
`macos/build/downloads`, which `make clean` keeps.

Check the embedded runtime with `make -C macos smoke`: it runs
`jetlink.server.main --list-backends` and opens a libusb context inside the
bundle, under the same stripped environment the app uses.

## Develop

```
make -C macos dev
```

That opens `Jetlink.xcodeproj` with `JETLINK_PYTHON` pointing at the repo's
`.venv`, so Xcode's Run needs no embedded runtime. Set the venv up once with
`pip install -e ".[ort,usb]"` plus tinygrad from git (the commit in
`Python/requirements-git.txt`; the PyPI wheel cannot parse the models).

`make -C macos project` regenerates the project from `project.yml` alone;
`make -C macos test` runs the Swift Testing suites. The generated
`Jetlink.xcodeproj` is committed, so `open macos/Jetlink.xcodeproj` works
without xcodegen installed.

## Sign, notarize, release

```
SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" DEVELOPMENT_TEAM=TEAMID make -C macos app
NOTARY_KEY_ID=... NOTARY_ISSUER_ID=... NOTARY_KEY_PATH=AuthKey_XXXX.p8 make -C macos notarize
SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" make -C macos dmg
```

`SIGN_IDENTITY` defaults to `-` (ad hoc), which is all a machine without a
Developer ID certificate can do; such a build runs locally but Gatekeeper will
not accept it on another Mac. The hardened runtime is on either way, and the
nested Python binaries are signed individually with
`Resources/python.entitlements`.

## Outputs

Everything lands in `macos/build/`: `Jetlink.app`, `python/` (the runtime
before it is copied into the bundle), `downloads/` (the interpreter tarball and
the wheels), `DerivedData/`, and from `make dmg` the `Jetlink-<version>.dmg`,
`Jetlink-<version>.zip` and `SHA256SUMS`. None of it is committed.

The app icon is generated once by `scripts/make-icon.swift` and its output is
committed under `Resources/Assets.xcassets`.
