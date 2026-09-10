# 40. Packaging: the Xcode project, embedded Python, signing, notarizing, DMG

**Owner: agent E.** Files you create: `macos/project.yml`, `macos/Makefile`,
`macos/README.md`, `macos/Resources/Info.plist`, `macos/Resources/Jetlink.entitlements`,
`macos/Resources/python.entitlements`, `macos/Resources/Assets.xcassets/` (AppIcon),
`macos/Python/requirements.txt`, `macos/Python/requirements-git.txt`,
`macos/Python/embed-python.sh`, `macos/scripts/build-app.sh`, `macos/scripts/sign.sh`,
`macos/scripts/notarize.sh`, `macos/scripts/make-dmg.sh`, `macos/scripts/make-icon.swift`,
`macos/scripts/check-version.sh`, `macos/.gitignore`, and the generated
`macos/Jetlink.xcodeproj/`. Read first: `01-contracts.md` sections 1, 2, 9;
`00-architecture.md` decisions D3, D4, D5. You need `brew install xcodegen libusb`.
Until agents C and D deliver sources, generate the project with a placeholder
`macos/Jetlink/App/JetlinkApp.swift` (an empty `App` with a `Text("Jetlink")`)
and the test target with one trivial test, so the pipeline is proven end to
end; the integrator swaps in the real sources.

## `macos/project.yml`

```yaml
name: Jetlink
options:
  bundleIdPrefix: io.zoompilot
  deploymentTarget: { macOS: "15.0" }
  xcodeVersion: "26.6"
  createIntermediateGroups: true
  generateEmptyDirectories: true
settings:
  base:
    SWIFT_VERSION: "6.0"
    SWIFT_STRICT_CONCURRENCY: complete
    MARKETING_VERSION: ${JETLINK_VERSION}          # set by Makefile from git describe; default 0.0.0
    CURRENT_PROJECT_VERSION: ${JETLINK_BUILD}      # default 1
    ENABLE_HARDENED_RUNTIME: YES
    CODE_SIGN_STYLE: Manual
    CODE_SIGN_IDENTITY: "-"                        # ad-hoc by default; release overrides on the command line
    DEVELOPMENT_TEAM: ""
    DEAD_CODE_STRIPPING: YES
    ARCHS: arm64
    ONLY_ACTIVE_ARCH: NO
targets:
  Jetlink:
    type: application
    platform: macOS
    sources:
      - path: Jetlink
      - path: Resources/Assets.xcassets
    info:
      path: Resources/Info.plist
      properties:
        CFBundleName: Jetlink
        CFBundleDisplayName: Jetlink
        CFBundleIdentifier: io.zoompilot.jetlink
        CFBundleShortVersionString: $(MARKETING_VERSION)
        CFBundleVersion: $(CURRENT_PROJECT_VERSION)
        LSMinimumSystemVersion: "15.0"
        LSApplicationCategoryType: public.app-category.developer-tools
        LSArchitecturePriority: [arm64]
        NSHumanReadableCopyright: "MIT License. Copyright 2026 Zeph Leggett."
        NSSupportsAutomaticTermination: false
        NSSupportsSuddenTermination: false
        NSPrincipalClass: NSApplication
        CFBundleIconName: AppIcon
    entitlements:
      path: Resources/Jetlink.entitlements
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: io.zoompilot.jetlink
        ENABLE_APP_SANDBOX: NO
        ENABLE_USER_SCRIPT_SANDBOXING: NO
    postBuildScripts:
      - name: Embed Python runtime
        script: |
          set -e
          SRC="$SRCROOT/build/python"
          DST="$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH/python"
          if [ ! -d "$SRC" ]; then echo "warning: no embedded Python at $SRC; run make python (the app will need JETLINK_PYTHON)"; exit 0; fi
          rm -rf "$DST"; mkdir -p "$(dirname "$DST")"
          rsync -a --delete "$SRC/" "$DST/"
        basedOnDependencyAnalysis: false
  JetlinkTests:
    type: bundle.unit-test
    platform: macOS
    sources: [JetlinkTests]
    dependencies: [{ target: Jetlink }]
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: io.zoompilot.jetlink.tests
schemes:
  Jetlink:
    build: { targets: { Jetlink: all, JetlinkTests: [test] } }
    run: { config: Debug, environmentVariables: { JETLINK_PYTHON: "$(JETLINK_PYTHON)" } }
    test: { config: Debug, targets: [JetlinkTests] }
```

Notes: `${JETLINK_VERSION}` and `${JETLINK_BUILD}` are xcodegen environment
substitutions; `make project` exports them. The post-build rsync copies the
prebuilt Python tree into the app on every build so Xcode's own "Run" has a
working runtime after `make python`. `xcodegen` warns about the `${...}`
usage only when unset; the Makefile always sets them.

## Entitlements

`Jetlink.entitlements`: an empty dict (hardened runtime needs the file to exist
to attach nothing). No sandbox keys.

`python.entitlements` (applied to `bin/python3.14`, `bin/python3`, and every
Mach-O under the prefix when signing):
```xml
<dict>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
</dict>
```
`allow-unsigned-executable-memory` is for ctypes/libffi closures;
`disable-library-validation` lets the interpreter load extension modules and
`libusb-1.0.dylib` even if a re-sign is ever incomplete. Never
`allow-dyld-environment-variables`, never `get-task-allow` in release.

## `macos/Python/requirements.txt` (hashed)

Generate once with `pip download` for `cp314` `macosx arm64` and
`pip hash`, then commit; the comment block at the top says how to regenerate.
Packages and versions from `01-contracts.md` section 1: onnxruntime, numpy,
onnx, libusb1, protobuf, ml_dtypes, typing_extensions, flatbuffers, packaging.
Wheel filenames for reference (2026-09-09):
```
onnxruntime-1.29.0-cp314-cp314-macosx_14_0_arm64.whl
numpy-2.5.3-cp314-cp314-macosx_11_0_arm64.whl
onnx-1.22.0-cp312-abi3-macosx_12_0_universal2.whl
libusb1-3.4.0-py3-none-any.whl
ml_dtypes-0.6.0-cp314-cp314-macosx_10_15_universal2.whl
protobuf-7.36.1-cp310-abi3-macosx_10_9_universal2.whl
typing_extensions-4.16.0-py3-none-any.whl
flatbuffers-25.12.19-py2.py3-none-any.whl
packaging-26.3-py3-none-any.whl
```
`requirements-git.txt`: one line,
`tinygrad @ git+https://github.com/sunnypilot/tinygrad@e837e367aac9e1a66e689f4f32ce20ca9367df13`. Installed
in a second `pip install --no-deps` step (hashes cannot cover a git source).

## `macos/Python/embed-python.sh`

`set -euo pipefail`. Inputs via environment with defaults: `PBS_TAG=20260901`,
`PBS_ASSET=cpython-3.14.7+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz`,
`PBS_SHA256=4632cb1a6edad9e73d3c81b6d2e69131637d995173e3e85005df14102b0592ba`,
`LIBUSB_DYLIB=/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib`,
`OUT=macos/build/python`, `DOWNLOADS=macos/build/downloads` (cached between runs).

Steps:
1. Download the tarball to `$DOWNLOADS` if missing; verify with `shasum -a 256`;
   abort on mismatch.
2. `rm -rf "$OUT"`; extract (the archive's top directory is `python/`) to `$OUT`.
3. `PY="$OUT/bin/python3"`. `"$PY" -m pip install --no-deps --require-hashes -r macos/Python/requirements.txt`
   then `"$PY" -m pip install --no-deps -r macos/Python/requirements-git.txt`
   then `"$PY" -m pip install --no-deps "$REPO_ROOT"` (the jetlink package,
   non-editable, so `importlib.metadata.version('jetlink')` works). If
   `git` is unavailable for the tinygrad step, fail with a clear message.
4. Copy `$LIBUSB_DYLIB` to `$OUT/lib/python3.14/site-packages/usb1/libusb-1.0.dylib`;
   `install_name_tool -id @loader_path/libusb-1.0.dylib` on the copy; `chmod 644`.
5. Prune, in this order and nothing else (each has been checked to be unused
   at runtime): `lib/python3.14/test`, `lib/python3.14/idlelib`,
   `lib/python3.14/tkinter`, `lib/python3.14/turtledemo`, `lib/python3.14/ensurepip`,
   `site-packages/pip*`, `site-packages/setuptools*`, `site-packages/wheel*`,
   `site-packages/numpy/**/tests`, `site-packages/onnx/backend/test/data`
   (50 MB of test vectors), `site-packages/onnx/test`, `share/`, `include/`,
   every `__pycache__`. Keep every `*.dist-info` (version lookups need them).
6. `"$PY" -m compileall -q -j 0 "$OUT/lib/python3.14"` (in-tree `__pycache__`,
   read at runtime with `PYTHONDONTWRITEBYTECODE=1`).
7. Sanity: `"$PY" -c "import jetlink, numpy, onnx, usb1, tinygrad; import importlib.metadata as m; print(m.version('onnxruntime'))"`
   and `"$PY" -m jetlink.server.main --list-backends` must print `ort` and
   `tinygrad` (the onnxruntime probe runs in a spawned child, so this also
   proves multiprocessing spawn works from the prefix). Both run with the
   runtime environment from `01-contracts.md` section 9 (`env -i PATH=... HOME=...`).
8. Write `$OUT/MANIFEST.json`: `{"python": "3.14.7", "pbs_tag", "pbs_sha256", "packages": {name: version from pip list --format json}, "tinygrad_commit": "e837e367aac9", "libusb": "1.0.30", "jetlink_git": "$(git rev-parse --short HEAD)", "built_at": ISO time}`.

Expected size after pruning: roughly 250 to 300 MB. Print `du -sh "$OUT"`.

## `macos/scripts/build-app.sh`

`JETLINK_VERSION` from `git describe --tags --always --dirty` stripped of a
leading `v` (fall back to `0.0.0`), `JETLINK_BUILD` from `git rev-list --count HEAD`.
`xcodegen generate --spec macos/project.yml`, then
`xcodebuild -project macos/Jetlink.xcodeproj -scheme Jetlink -configuration Release -derivedDataPath macos/build/DerivedData build CODE_SIGN_IDENTITY="${SIGN_IDENTITY:--}" ${DEVELOPMENT_TEAM:+DEVELOPMENT_TEAM=$DEVELOPMENT_TEAM} CODE_SIGNING_ALLOWED=YES`,
then copy the product to `macos/build/Jetlink.app`. Print the path.

## `macos/scripts/sign.sh`

Arguments: the app path; `SIGN_IDENTITY` (default `-`). With `-` it ad-hoc
signs (no timestamp, no hardened runtime requirement, but keep `--options
runtime` so the local build behaves like the release). Steps, inside out:
1. Find Mach-O files under `Contents/Resources/python`: `find ... -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \)`
   then filter with `file -b` matching `Mach-O`. Sign each:
   `codesign --force --options runtime --timestamp --entitlements macos/Resources/python.entitlements --sign "$SIGN_IDENTITY" "$f"`
   (omit `--timestamp` for `-`). Do the executables in `bin/` last among them.
2. Sign the app: `codesign --force --options runtime --timestamp --entitlements macos/Resources/Jetlink.entitlements --sign "$SIGN_IDENTITY" macos/build/Jetlink.app`.
3. Verify: `codesign --verify --deep --strict --verbose=2 Jetlink.app` and
   `codesign -d --entitlements :- Jetlink.app/Contents/Resources/python/bin/python3.14`
   showing the two entitlements. With a real identity also
   `spctl --assess --type execute --verbose Jetlink.app` (expected to fail
   before notarization with "Unnotarized Developer ID"; print and continue).

## `macos/scripts/notarize.sh`

Requires `NOTARY_KEY_ID`, `NOTARY_ISSUER_ID`, `NOTARY_KEY_PATH` (an
AuthKey `.p8`). `ditto -c -k --keepParent Jetlink.app Jetlink.zip`;
`xcrun notarytool submit Jetlink.zip --key "$NOTARY_KEY_PATH" --key-id ... --issuer ... --wait --timeout 30m`;
on anything but `Accepted`, `xcrun notarytool log <id>` to stderr and exit 1;
`xcrun stapler staple Jetlink.app`; `spctl --assess --type execute` must now pass.

## `macos/scripts/make-dmg.sh`

No third-party tool. A staging dir with `Jetlink.app` and a symlink
`Applications -> /Applications`; `hdiutil create -volname Jetlink -srcfolder stage -ov -format UDZO Jetlink-<version>.dmg`;
sign the DMG with the same identity; notarize it (`notarytool submit` the dmg,
`stapler staple` it) when credentials are present. Also produce
`Jetlink-<version>.zip` (`ditto`) of the stapled app and a `SHA256SUMS` file.

## `macos/scripts/make-icon.swift`

A `swift` script (run with `swift macos/scripts/make-icon.swift out.png`) that
draws a 1024×1024 icon with CoreGraphics: a rounded square in the system's
blue (`#1E6FD9`) with a white cable-connector glyph made of two rounded
rectangles and a line, nothing fancy; then `sips -z` to produce 16, 32, 64,
128, 256, 512, 1024 into `Assets.xcassets/AppIcon.appiconset/` with a
`Contents.json` listing the standard 10 macOS entries (1x and 2x per size).
Committed output, so the script runs once.

## `macos/scripts/check-version.sh`

Given a tag `vX.Y.Z`, verify `pyproject.toml`'s `version = "X.Y.Z"` matches;
exit 1 with both values otherwise. Used by the release workflow.

## `macos/Makefile`

```
project   xcodegen generate (exports JETLINK_VERSION/JETLINK_BUILD)
python    Python/embed-python.sh
app       project + python + scripts/build-app.sh + scripts/sign.sh (ad-hoc unless SIGN_IDENTITY set)
test      xcodebuild test -scheme Jetlink -destination 'platform=macOS'
notarize  scripts/notarize.sh build/Jetlink.app
dmg       scripts/make-dmg.sh
dev       open Jetlink.xcodeproj with JETLINK_PYTHON defaulting to ../.venv/bin/python
clean     rm -rf build/ (keeps build/downloads)
smoke     runs build/Jetlink.app/Contents/Resources/python/bin/python3 -m jetlink.server.main --list-backends under the runtime environment
```
`.PHONY` all; every recipe `set -euo pipefail`.

## `macos/.gitignore`

`build/` (except `build/downloads` is inside it, so just `build/`),
`DerivedData/`, `*.xcuserstate`, `xcuserdata/`.

## `macos/README.md`

How to build (`brew install xcodegen libusb`, `make app`), how to develop
(`make dev`, which opens Xcode with `JETLINK_PYTHON` pointing at the repo's
`.venv` so no embedding is needed; the venv must have `pip install -e
".[ort,usb]"` plus tinygrad from git), how to sign and notarize locally
(environment variables), where the outputs land. Ten lines each, not essays.

## Gotchas to write into the scripts as comments

- The signed bundle must never be modified at runtime, hence
  `PYTHONDONTWRITEBYTECODE=1` (agent C sets it) and the in-tree `compileall`.
- `codesign --deep` is not used; nested Mach-O files are signed explicitly,
  which is what the notary service checks.
- The ORT worker is a `multiprocessing` spawn: the interpreter path must be a
  real file, not a symlink to a Homebrew cellar (python-build-standalone's
  `bin/python3` is a symlink to `python3.14` inside the prefix; that is fine).
- Homebrew's dylib has an absolute `LC_ID_DYLIB`; ctypes loads it by path so
  the id does not matter, but `install_name_tool -id` keeps `otool -L` honest.
- `hdiutil` occasionally fails with "resource busy" on CI; retry once.

## Report back

`du -sh macos/build/python`, the output of `make smoke`, `codesign --verify`
output, and whether `spctl` accepted the ad-hoc build (it will not; say so).
