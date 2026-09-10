# 00. Architecture and decisions

Read this before any workstream file. It explains what exists today, what the
app adds, where each piece lives, and why. Decisions are final unless the code
proves them impossible; then say so in your report.

## What exists today (do not rediscover this)

Jetlink is a Python package. The comma (running the zoompilot `jetson-trt`
fork) is a USB gadget; the server is the USB host. The comma asks the server
for a model by its ONNX SHA-256, uploads the ONNX if the server lacks it, waits
while the server builds an engine and loads it, then streams frames at 20 Hz.

Key server facts, all in `jetlink/server/`:

- `main.py` is the entry point (`jetlink-server` / `python -m jetlink.server.main`).
  `--backend auto|trt|ort|tinygrad`, `--device`, `--transport usb|tcp|ffs`,
  `--cache DIR`, `--build ONNX` (build and exit), `--list-backends`.
  `_serve()` loops: open a transport (or wait 2 s), run a `Session` until the
  link drops, repeat. One `EngineHost` outlives every session.
- `session.py`: `EngineHost` owns the one loaded engine and the one build job
  in flight (`Job` with `state` building|ready|failed and `load_only`). It
  reports progress through `self.session.progress(stage, frac, msg)` and
  completion through `session.engine_update()`. `Session` is one client
  connection; `_infer` is the hot path and must stay allocation-free.
- `cache.py`: `EngineCache(root, backend)`. Layout under the root:
  `engines/<sha16>.<backend tag><suffix>` (artifact), `engines/<sha16>.<tag>.json`
  (sidecar with `spec`, `backend`, `device`, `built_at`, `build_seconds`, and a
  runtime version under `onnxruntime` / `tinygrad` / `trt_version`),
  `models/<sha16>.onnx` (the uploaded model), `last-loaded.json` (what to
  preload next start). `inventory()` lists only the current backend's artifacts.
- `backends/`: `trt` (Jetson, `.plan`), `ort` (CoreML on a Mac, `.ortcache`
  directory), `tinygrad` (Metal, `.pkl`). On a Mac `auto` picks CoreML on the
  GPU: 43 ms a frame, but a 9 minute compile every time a process creates the
  session, so the server keeps the engine loaded for its whole life. tinygrad
  loads in 1 s but runs 66 ms a frame on an M1 Pro (over the 50 ms budget).
- Two runtime constraints found by crashing: tinygrad is driven from a single
  owner thread (`backends/tinygrad/owner.py`), and onnxruntime lives in a
  spawned worker process (`backends/ort/worker.py`) because it holds the GIL
  for the whole session creation. The server process never imports onnxruntime.
- `platform.py`: `default_cache_dir()` is `JETLINK_CACHE`, else the Jetson's
  mount, else `~/Library/Caches/jetlink` on a Mac. `scripts/run-mac.sh`
  overrides it to `models_cache/` beside the checkout and wraps the server in
  `caffeinate -s`.

Model registry facts, currently implemented only in the fork
(`sunnypilot-jetson-trt/openpilot/sunnypilot/accelerators/jetlink/helpers.py`
and `lfs.py`, both depend on openpilot and cannot be imported here):

- The comma's model list is sunnypilot's chestnut catalog:
  `https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/driving_models_chestnut_v25.json`.
  `bundles[]` with `display_name`, `short_name`, `ref` (a 40-hex comma
  openpilot commit), `index`, `build_time`, `minimum_selector_version` ("19"
  today, as a string), `is_big`. 13 entries as of 2026-09-09; a copy is in
  `fixtures/catalog_chestnut_v25.json`.
- The ONNX for a catalog entry is the git-lfs object at that commit:
  `https://raw.githubusercontent.com/commaai/openpilot/<ref>/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx`
  returns the 134-byte LFS pointer text (`oid sha256:<hex>`, `size <n>`) for
  every catalog ref, merged or not. The oid is the SHA-256 the comma asks the
  server for. Pointers never change for a given ref; cache them forever.
- The bytes come from an LFS batch API, first server that has the object wins:
  `https://gitlab.com/commaai/openpilot-lfs.git/info/lfs` (everything),
  then `https://huggingface.co/commaai/openpilot-lfs.git/info/lfs` (current
  master only). POST `{operation: download, transfers: [basic], objects: [{oid, size}]}`
  with `Accept`/`Content-Type: application/vnd.git-lfs+json`; the reply's
  `objects[0].actions.download.href` is a short-lived URL; an `error` object
  means "not here". Fixtures: `fixtures/lfs_batch_response.json`,
  `fixtures/lfs_batch_missing.json`.
- The fork's default big model is ref `f877d7a0ccc3cce943c76e285214c020cd65c899`
  (BMRLNAP Model v4, oid `a086d5249fc308bb…`, 765953504 bytes).

Python runtime facts for packaging:

- The bench venv is Python 3.14.5 with onnxruntime 1.29.0, numpy 2.5.3,
  onnx 1.22.0, libusb1 3.4.0, tinygrad 0.14.0. All have arm64 macOS wheels for
  CPython 3.14.
- **tinygrad from PyPI (0.14.0) cannot parse the real models**: its
  `OnnxPBParser` has no `org.tinygrad` domain, and the exported models carry
  `org.tinygrad` nodes. This is why four tests in `tests/test_tinygrad_backend.py`
  error in the current venv with `'org.tinygrad' is not a valid Domain`. The
  measured backend used a git checkout at commit `e837e367aac9` (sunnypilot/tinygrad master, full sha `e837e367aac9e1a66e689f4f32ce20ca9367df13`), which has the
  domain. The app pins tinygrad to that commit from git, never to the PyPI wheel.
- python-libusb1 loads the native library by trying, first,
  `<site-packages>/usb1/libusb-1.0.dylib`, then the Homebrew path. Dropping
  the dylib into the `usb1/` package directory is the whole bundling story.
  Homebrew's libusb is 1.0.30.

## What the app adds

Three things, in order of value to the user:

1. **Always on, right model loaded.** The comma's 9 minute wait on a Mac is
   CoreML compiling. The server already keeps the engine resident; the app
   keeps the server alive, keeps the Mac awake, restarts it if it dies, and
   shows whether the comma is connected and how the frames are doing.
2. **Models ahead of time.** Pick a model from the same list the comma shows,
   download it over the Mac's network (faster than the comma's LTE plus the USB
   upload), prepare it, and have it loaded before the drive. The comma's
   request then hits `ready`.
3. **Cache management.** See what is on disk (a CoreML artifact is 5.5 GB),
   delete downloads while keeping prepared engines, or delete engines.

## Decisions

**D1. Management logic is Python, cross-platform, with a CLI.** New package
`jetlink/registry/` (catalog, pointers, LFS, local imports, inventory) and
new module `jetlink/server/control.py` (a local control channel the server
speaks). `jetlink-models` is the CLI over the registry. The Jetson image gets
both for free (the Dockerfile copies `jetlink/`), so a Jetson can prefetch a
model over its own network and a Linux laptop gets the same commands. The fork
may later replace its `helpers.fetch_pointer`/`lfs.py` with imports from here;
that is not part of this work.

**D2. One Python process, owned by the app, does all GPU and network work.**
The app launches `python -m jetlink.server.main` with `--control-socket` and
`--parent-pid`, connects to the socket, and drives everything through JSON
commands. Downloads, imports, builds and loads all happen inside the server
process, so there is never a second process fighting over the cache or the
GPU. "Prepare" from the app is the same code path as the comma's ENGINE_REQ:
build if needed, then load and keep loaded.

**D3. Embedded, relocatable CPython; no system Python, no Homebrew.**
`Jetlink.app/Contents/Resources/python/` is a python-build-standalone 3.14.7
install (stripped tarball, checksum pinned) with the pinned wheels installed
into its site-packages, the `jetlink` package installed with `pip install
<repo>` (so `importlib.metadata` sees it), tinygrad from git at the pinned
commit, and Homebrew's `libusb-1.0.dylib` copied into `site-packages/usb1/`.
Precompiled with `compileall`; run with `PYTHONDONTWRITEBYTECODE=1` so the
signed bundle is never modified at runtime. Every Mach-O inside is signed
individually with the hardened runtime. Not PyInstaller or py2app: the ORT
worker is a `multiprocessing` spawn that re-imports `__main__`, which frozen
apps break.

**D4. The Xcode project is generated by xcodegen from `macos/project.yml`.**
Hand-maintaining a `.pbxproj` across parallel agents is how merges die. The
generated `Jetlink.xcodeproj` is committed too, so `open` works without
installing anything; `make project` regenerates it and CI checks it is fresh.

**D5. No App Sandbox. Developer ID signed and notarized, distributed via
GitHub Releases.** libusb needs the device, the server writes a
user-chosen cache folder, and the app spawns an interpreter tree from its
Resources. The App Store is not a goal. Hardened runtime on, with the two
entitlements CPython plus ctypes needs
(`com.apple.security.cs.allow-unsigned-executable-memory`,
`com.apple.security.cs.disable-library-validation`) on the Python binaries.

**D6. Regular app plus a menu bar extra.** A normal window (Status, Models,
Logs in a sidebar) and a `MenuBarExtra` with the status line and Start/Stop,
so closing the window keeps the server running visibly. Quitting the app stops
the server; that is what a user expects from the thing that owns it.

**D7. Sleep is prevented natively.** `IOPMAssertionCreateWithName` with
`PreventUserIdleSystemSleep` while serving and on AC power (the same rule as
`caffeinate -s`), re-evaluated on power source changes. No `caffeinate`
subprocess.

**D8. Cache root default is `~/Library/Application Support/Jetlink/cache/`.**
Not `~/Library/Caches`: a 5.5 GB artifact that took 9 minutes must not be
deleted by a cleaner tool. Changeable in Settings (existing users point it at
their `models_cache/`). The layout inside is `EngineCache`'s, unchanged, plus a
`registry/` directory the Python registry owns.

**D9. Swift 6 language mode, macOS 15 deployment target, no dependencies.**
Sparkle auto-updates are a follow-up; the release pipeline produces the
artifacts an appcast would need.

**D10. The wire protocol, queues, backends and the comma-side client do not
change.** Nothing in this work touches `protocol.py`, `queues.py`, `client.py`,
`transport/`, or `backends/`. Hello and telemetry stay as they are.

## Repository layout after this work

```
jetlink/
  registry/                     NEW  A: catalog.py, lfs.py, __init__.py (Registry), cli.py, __main__.py
  server/control.py             NEW  B: the control channel
  server/session.py             EDIT B: listeners, frame stats, snapshot, public unload
  server/main.py                EDIT B: --control-socket, --parent-pid, SIGTERM, link events
pyproject.toml                  EDIT A: jetlink-models entry point
tests/
  fixtures/                     NEW  A: copied from plans/macos-app/fixtures
  test_registry.py              NEW  A
  test_control.py               NEW  B
macos/
  README.md                     E: how to build and run in development
  Makefile                      E: project, python, app, sign, notarize, dmg, test, clean
  project.yml                   E: xcodegen spec
  Jetlink.xcodeproj/            E: generated, committed
  Jetlink/                      C and D: app sources (layout in 20 and 30)
  JetlinkTests/                 C and D: Swift Testing targets
  Python/
    requirements.txt            E: pinned wheels with hashes
    requirements-git.txt        E: tinygrad at the pinned commit
    embed-python.sh             E: builds build/python from the tarball and wheels
  Resources/
    Jetlink.entitlements        E: the app's entitlements (empty set, hardened runtime)
    python.entitlements         E: the Python binaries' entitlements
    Assets.xcassets/            E: AppIcon
    Info.plist                  E
  scripts/
    build-app.sh sign.sh notarize.sh make-dmg.sh make-icon.swift check-version.sh   E
.github/workflows/
  ci.yml release.yml            F
docs/macos-app.md               G: the user guide
docs/models.md                  G: the jetlink-models CLI on every platform
```

## Process model and data flow

```
 Jetlink.app (Swift)                              python -m jetlink.server.main
 ┌──────────────────────────┐   spawn, SIGINT     ┌────────────────────────────────┐
 │ ServerProcess ───────────┼────────────────────►│ main.py  (--parent-pid, --control-socket)
 │   stderr pump ◄──────────┼── stderr (logs) ────┤   logging → stderr
 │ ControlClient ◄──────────┼── unix socket ──────┤ control.py  ControlServer
 │   AsyncStream<Event>     │   JSON lines        │   ├─ subscribes to EngineHost.emit
 │   send(cmd) → reply      │                     │   ├─ registry: catalog, download, import
 │ ServerStore / ModelStore │                     │   ├─ inventory: EngineCache + sidecars
 │ SwiftUI views            │                     │   └─ 1 Hz stats from Session frames
 └──────────────────────────┘                     │ session.py  EngineHost ── Session ── USB ── comma
                                                  │ backends/ort worker (spawned child)
                                                  └────────────────────────────────┘
```

- The app never reads or writes the cache directly. It asks the server
  (`inventory`, `forget`) and renders what comes back. The one exception is
  "Reveal in Finder", which only needs the path the server reported.
- Progress of a build/load reaches the app as `engine` events with
  `stage`/`frac`/`msg`, the same values the comma sees as `PROGRESS`.
- The app's Logs view is the server's stderr, line by line, also teed to
  `~/Library/Logs/Jetlink/server.log`.

## Non-goals for this delivery

- Sparkle or any auto-update; App Store; sandboxing.
- Changing the comma-side fork or the wire protocol.
- Telemetry on a Mac (the sensors need privileges; the comma is told nothing,
  as today).
- Running more than one engine at a time, or more than one server.
- A Windows or Linux GUI. Those platforms get the CLI and the control channel;
  a GUI there is a separate project.
- Support for Intel Macs. The embedded runtime is arm64 only; `LSArchitecturePriority` lists arm64 and the release notes say so.
