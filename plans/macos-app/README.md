# Jetlink for Mac: implementation plan

A native macOS app that runs the Jetlink inference server, shows what the comma
sees, and manages the large driving models: download them from sunnypilot's
big-model catalog ahead of time, prepare (compile) them for the Mac's backend,
and keep the right one loaded so the comma drives with it the moment it plugs in.

These files are written for implementation agents working in parallel. Each
workstream file is self-contained: read `00-architecture.md` and
`01-contracts.md` first (everyone), then your own file. Do not read the whole
codebase before starting; each file names exactly which existing files matter.

## The one rule that shapes everything

**Anything that is not the SwiftUI shell goes in the Python package and works
on every platform.** The catalog fetch, the LFS download, the cache inventory,
the control channel, parent-process watching, clean shutdown on SIGTERM: all of
it lives in `jetlink/` (Python, stdlib only) with a CLI, so a Jetson, a Linux
laptop, a Windows box and the Mac terminal get the same features. The Swift app
is a thin client of those. If you find yourself writing model, network or cache
logic in Swift, stop: it belongs in Python behind a control-channel command.

## Files

| File | Who reads it | What it is |
| --- | --- | --- |
| `00-architecture.md` | everyone | Decisions and why, repo layout, process model, data flow, non-goals |
| `01-contracts.md` | everyone | The shared interfaces: control protocol, CLI flags, paths, settings keys, Swift API shapes, pinned versions |
| `10-python-registry.md` | agent A | `jetlink/registry/`: catalog, LFS pointer, download, inventory, local models, `jetlink-models` CLI |
| `11-python-control-channel.md` | agent B | `jetlink/server/control.py` plus the hooks in `session.py` and `main.py`; `--control-socket`, `--parent-pid`, SIGTERM |
| `20-swift-core.md` | agent C | Non-UI Swift: embedded Python lookup, server process, control client, stores, sleep assertion, settings, logs |
| `30-swiftui-views.md` | agent D | The windows, the menu bar extra, settings; exact strings and layout |
| `40-packaging-and-signing.md` | agent E | xcodegen project, Info.plist, entitlements, embedding Python and wheels, libusb, signing, notarizing, DMG, Makefile |
| `50-ci-and-releases.md` | agent F | GitHub Actions: lint and tests for Python and Swift, app build, signed and notarized releases on tags, Jetson image to GHCR |
| `60-docs.md` | agent G | User guide for the app, README and platform docs updates, the CLI on other platforms |
| `70-integration-and-qa.md` | the integrator | Merge order, end-to-end smoke tests, acceptance checklist, the bench test with a real comma |
| `fixtures/` | A, B, C | Offline copies of the live catalog, an LFS pointer, and LFS batch responses, for tests |

## Parallelism and dependencies

```
A  python registry ─────────┐
B  python control channel ──┼──► H integration ──► bench test
C  swift core ──────────────┤        ▲
D  swiftui views ───────────┤        │
E  packaging ───────────────┤        │
F  ci/release ──────────────┘        │
G  docs ─────────────────────────────┘
```

A and B are independent if B codes against the `Registry` class signature in
`01-contracts.md` (B may stub it until A lands). C codes against the control
protocol in `01-contracts.md` and can be tested with the JSON fixtures before B
exists. D codes against the Swift types in `01-contracts.md`, using preview
mocks, and is wired to C's real stores by the integrator. E needs nothing from
the others to produce a project that builds an empty app; it needs C and D to
build the real one. F needs E's scripts and A/B's tests.

Every agent: keep to the files your workstream names. If you must touch a file
another workstream owns, make the smallest change and say so in your report.

## Conventions every agent follows

**Python**
- Two-space indentation, 160 columns, `from __future__ import annotations`,
  the license header every existing module carries (copy it verbatim from
  `jetlink/spec.py`). `ruff check .` must pass with the repo's `pyproject.toml`.
- Stdlib only. The package's one dependency is numpy; do not add `requests`
  or anything else. `urllib.request` does everything the registry needs.
- Nothing may import an inference runtime at module scope, and nothing in the
  hot path (`Session._infer`) may allocate, log or take a lock it does not
  already take. Read the docstring at the top of `jetlink/server/session.py`.
- Tests in `tests/`, pytest, offline: patch `urllib.request.urlopen` or pass a
  fetch function; never hit the network in a test. Fixtures from
  `plans/macos-app/fixtures/` are copied into `tests/fixtures/`.
- Commit messages: terse `area: subject`, no trailer, no em dash.

**Swift**
- Swift 6 language mode, macOS 15.0 deployment target, Xcode 26.6. No third-party
  packages. Apple frameworks only: SwiftUI, AppKit, Foundation, Network,
  CryptoKit, IOKit, ServiceManagement, OSLog, UniformTypeIdentifiers, Swift
  Testing.
- Observable state lives in `@MainActor @Observable final class` stores. Work
  off the main actor happens in `Task.detached` or an `actor`; results come
  back through `await MainActor.run { }` or `for await` on an `AsyncStream`.
  Never `nonisolated(unsafe)`, never `@unchecked Sendable` without a comment
  saying which invariant makes it safe.
- Every user-visible string is plain English in sentence case, as the macOS
  Human Interface Guidelines put it. No emoji. No exclamation marks.
- Log with `os.Logger(subsystem: "io.zoompilot.jetlink", category: "...")`;
  never `print` in app code.
- Unit tests with Swift Testing (`import Testing`, `@Test`, `#expect`).

**Both**
- No em dashes anywhere: code, comments, docs, commit messages, UI strings.
- Do not rename, move or reformat existing files beyond what your task needs.
- When something in these plans turns out to be wrong against the code you
  find, follow the code, note the discrepancy in your report, and keep the
  contract in `01-contracts.md` intact unless you coordinate a change.

## Definition of done

1. `make app` in `macos/` produces `macos/build/Jetlink.app` that launches on a
   clean Apple-silicon Mac with no Homebrew, no Python and no Xcode installed.
2. The app starts the server, shows "Waiting for comma", lists the catalog,
   downloads a model, prepares it, and shows it loaded; a comma plugged into a
   USB-A port then connects and the Status view shows frames at 20 Hz.
3. `pytest` and `ruff check .` pass on the Python side; `xcodebuild test`
   passes on the Swift side; both run in GitHub Actions on every push.
4. Pushing a `v*` tag produces a GitHub release with a notarized DMG and zip
   (when signing secrets are configured; an ad-hoc signed zip otherwise), the
   Python sdist and wheel, and a Jetson image on GHCR.
5. `jetlink-models list` and `jetlink-models fetch <ref>` work on a Jetson and a
   Linux machine with only the Python package installed.

After the build, these plan files can be deleted or moved under `docs/`; the
durable documentation is what `60-docs.md` produces.
