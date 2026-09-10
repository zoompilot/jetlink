# 50. CI and releases on GitHub

**Owner: agent F.** Files you create: `.github/workflows/ci.yml`,
`.github/workflows/release.yml`, `.github/dependabot.yml`,
`docs/releasing.md` (edit: add the tag-to-release flow). Read first:
`40-packaging-and-signing.md` (you call its scripts), `01-contracts.md`
section 1, and the existing `docs/releasing.md`. The repo's remote is
`github.com/zoompilot/jetlink`; there is no `.github/` today.

## Goals

1. Every push and pull request: lint and test the Python package on Linux
   and macOS, build and test the Swift app, build the full app bundle with the
   embedded runtime and run a smoke test against it. Fast paths first, so a
   ruff failure is reported in a minute.
2. Every `v*` tag: a GitHub release with a notarized `Jetlink-X.Y.Z.dmg` and
   `.zip` (or an ad-hoc signed zip when signing secrets are absent, clearly
   labelled), `SHA256SUMS`, the Python sdist and wheel, generated release
   notes, and a Jetson Docker image pushed to GHCR.
3. Nothing in CI depends on the runner having Homebrew packages beyond
   `xcodegen` and `libusb`, which the workflow installs.

## Runners

- Linux: `ubuntu-24.04`.
- macOS: `macos-26` (Apple silicon; Xcode 26.6 is the default there as of
  2026-07-21). Pin Xcode explicitly anyway: `sudo xcode-select -s /Applications/Xcode_26.6.app`
  guarded by an `ls /Applications | grep Xcode` step that prints what exists,
  so a runner image change fails loudly with the list.

## `ci.yml`

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:
concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }
permissions: { contents: read }
jobs:
  python-lint:
    runs-on: ubuntu-24.04
    steps: checkout; setup-python 3.12; pip install ruff==<the venv's, 0.16.6>; ruff check .; ruff format --check . (only if the repo already passes; it uses quote-style preserve; check first and drop the format step if it fails on untouched files)
  python-test-linux:
    runs-on: ubuntu-24.04
    strategy: { matrix: { python: ["3.10", "3.12"] } }
    steps: checkout; setup-python; pip install -e ".[dev]"; pytest -q
      # no runtimes: the backend tests importorskip; the registry and control tests run
  python-test-macos:
    runs-on: macos-26
    steps: checkout; setup-python 3.14; pip install -e ".[dev,ort,usb]"; pip install "tinygrad @ git+https://github.com/sunnypilot/tinygrad@e837e367aac9e1a66e689f4f32ce20ca9367df13"; brew install libusb; pytest -q
      # runs test_ort_backend (CPU provider) and test_tinygrad_backend (CPU device) on a real runtime
  swift:
    runs-on: macos-26
    steps: checkout; select Xcode; brew install xcodegen; make -C macos project; git diff --exit-code macos/Jetlink.xcodeproj (the committed project must be fresh); make -C macos test
  app:
    runs-on: macos-26
    needs: [python-lint]
    steps:
      checkout; select Xcode; brew install xcodegen libusb
      actions/cache: path macos/build/downloads, key pbs-${{ hashFiles('macos/Python/embed-python.sh') }}
      actions/cache: path ~/Library/Caches/pip, key pip-${{ hashFiles('macos/Python/requirements*.txt') }}
      make -C macos python
      make -C macos app          # ad-hoc signed
      make -C macos smoke
      python smoke against the embedded runtime (below)
      actions/upload-artifact: Jetlink-unsigned.zip (ditto of the app), retention-days 7
  swift-format:
    runs-on: macos-26
    continue-on-error: true
    steps: checkout; select Xcode; xcrun swift-format lint --strict --recursive macos/Jetlink macos/JetlinkTests
```

### The embedded-runtime smoke test (in `app`)

A script `macos/scripts/smoke-control.py`, run with the embedded interpreter
under the runtime environment from `01-contracts.md` section 9:
1. Write `tests/tiny_model.py`'s ONNX to a temp cache's `models/` under its
   sha16 name (import `tests.tiny_model` with the repo on `sys.path` for the
   smoke only).
2. Start `python -m jetlink.server.main --backend ort --device cpu --transport tcp --port 0`
   (add `--port 0` support? The TCP listener already accepts any port;
   `0` binds an ephemeral one, and the log line says which; the smoke does
   not need the port) `--cache <tmp> --control-socket <tmp>/c.sock --parent-pid <self>`.
3. Connect to the socket, assert the six on-connect events, send `prepare`
   for the tiny model's sha, wait for `engine ready` (60 s), assert
   `inventory` lists one artifact with `current: true`, send `shutdown`,
   assert exit code 0 within 15 s.
This proves: the bundle's Python runs, onnxruntime spawns its worker from the
prefix, the control channel works, and a build writes the cache. It takes
under 30 s.

## `release.yml`

```yaml
name: Release
on:
  push: { tags: ["v*"] }
permissions: { contents: write, packages: write, id-token: write }
jobs:
  check:
    runs-on: ubuntu-24.04
    steps: checkout; macos/scripts/check-version.sh "${GITHUB_REF_NAME}"
  python-dist:
    needs: check
    runs-on: ubuntu-24.04
    steps: checkout; setup-python 3.12; pip install build; python -m build; upload-artifact dist/
  macos-app:
    needs: check
    runs-on: macos-26
    env:
      HAVE_SIGNING: ${{ secrets.MACOS_CERTIFICATE_P12_BASE64 != '' && secrets.NOTARY_KEY_ID != '' }}
    steps:
      checkout (fetch-depth 0, for git describe); select Xcode; brew install xcodegen libusb
      caches as in ci.yml
      - name: Import signing certificate
        if: env.HAVE_SIGNING == 'true'
        run: |
          KEYCHAIN=$RUNNER_TEMP/build.keychain-db
          security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
          security set-keychain-settings -lut 21600 "$KEYCHAIN"
          security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
          echo "$MACOS_CERTIFICATE_P12_BASE64" | base64 --decode > $RUNNER_TEMP/cert.p12
          security import $RUNNER_TEMP/cert.p12 -P "$MACOS_CERTIFICATE_PASSWORD" -A -t cert -f pkcs12 -k "$KEYCHAIN"
          security set-key-partition-list -S apple-tool:,apple: -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN"
          security list-keychains -d user -s "$KEYCHAIN" login.keychain-db
          echo "SIGN_IDENTITY=$(security find-identity -v -p codesigning "$KEYCHAIN" | grep 'Developer ID Application' | head -1 | sed -E 's/.*"(.*)"/\1/')" >> $GITHUB_ENV
          echo "$NOTARY_PRIVATE_KEY_P8_BASE64" | base64 --decode > $RUNNER_TEMP/AuthKey.p8
          echo "NOTARY_KEY_PATH=$RUNNER_TEMP/AuthKey.p8" >> $GITHUB_ENV
        env: (the secrets)
      - make -C macos python
      - make -C macos app         # SIGN_IDENTITY set → Developer ID; otherwise ad-hoc
      - make -C macos notarize    # if HAVE_SIGNING
      - make -C macos dmg         # dmg + zip + SHA256SUMS; when unsigned, only the zip, named Jetlink-X.Y.Z-unsigned.zip
      - always: security delete-keychain "$KEYCHAIN" (if created)
      - upload-artifact macos/build/dist/*
  docker-jetson:
    needs: check
    runs-on: ubuntu-24.04
    steps:
      checkout; docker/setup-qemu-action; docker/setup-buildx-action; docker/login-action to ghcr.io with GITHUB_TOKEN
      docker/metadata-action for ghcr.io/zoompilot/jetlink with tags type=semver
      docker/build-push-action: file docker/Dockerfile, platforms linux/arm64, push true, cache-from/to type=gha
      # Base image nvcr.io/nvidia/l4t-jetpack:r36.4.0 is public and several GB; under QEMU pip installs
      # numpy (aarch64 wheel exists) and pure-Python packages, no compilation. Budget 30 min; timeout-minutes: 60.
      # A failure here must not block the macOS release: continue-on-error: true, and the release notes say so.
  release:
    needs: [python-dist, macos-app, docker-jetson]
    if: always() && needs.python-dist.result == 'success' && needs.macos-app.result == 'success'
    runs-on: ubuntu-24.04
    steps:
      download-artifact (all)
      gh release create "$GITHUB_REF_NAME" --generate-notes --title "Jetlink $GITHUB_REF_NAME" \
         $( [[ "$GITHUB_REF_NAME" == *-* ]] && echo --prerelease ) artifacts/**/*
      append to the notes: the image tag ghcr.io/zoompilot/jetlink:X.Y.Z (or "image build failed" when docker-jetson did not succeed),
      and for an unsigned build the Gatekeeper sentence from docs/macos-app.md.
```

### Secrets to create (document in `docs/releasing.md`)

| Secret | What |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | Developer ID Application certificate with private key, exported from Keychain Access as .p12, base64 |
| `MACOS_CERTIFICATE_PASSWORD` | the .p12 password |
| `KEYCHAIN_PASSWORD` | any random string for the temporary keychain |
| `APPLE_TEAM_ID` | 10-character team id (also used as `DEVELOPMENT_TEAM`) |
| `NOTARY_KEY_ID`, `NOTARY_ISSUER_ID`, `NOTARY_PRIVATE_KEY_P8_BASE64` | App Store Connect API key (Team key, Developer role) for notarytool |

Without them the release still happens, unsigned, so contributors' forks work.
The workflow prints which mode it ran in.

## `dependabot.yml`

Weekly updates for `github-actions` only. Python and Swift pins are
deliberate (measured versions); do not add ecosystems for them.

## `docs/releasing.md` additions

A "Cutting a release" section: bump `pyproject.toml` version, commit, tag
`vX.Y.Z`, push the tag, wait for the Release workflow, check the release
page; how to install the DMG; the GHCR image and how the Jetson service can
pin it (`docker pull ghcr.io/zoompilot/jetlink:X.Y.Z`, then the existing
`server.env` image-id step). Keep the existing rollback text.

## Local verification before reporting

- `act` is not required. Instead run each script the workflows call, locally:
  `make -C macos python app smoke test`, the smoke-control script, and
  `python -m build`.
- Validate the YAML with `gh workflow view` after pushing to a branch, or at
  least `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`
  (PyYAML is in the venv? If not, `ruby -ryaml -e 'YAML.load_file(ARGV[0])'`).
- Push the branch and confirm `ci.yml` is green on GitHub before merging; the
  first run will show whether `macos-26` and `Xcode_26.6.app` exist as pinned.

## Report back

Links to the first green CI run, the artifact size, the wall time of each job,
and any step that needed a change from this plan.
