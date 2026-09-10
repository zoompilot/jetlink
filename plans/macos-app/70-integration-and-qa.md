# 70. Integration and QA

**Owner: the integrator (the orchestrating agent, or one agent it delegates
to after all workstreams report).** This is the order to merge, the checks to
run at each step, and the acceptance list. Nothing here is optional.

## Merge order

1. **A (registry) and B (control channel).** Merge A first, then B (B's tests
   need `Registry`). Run `ruff check .` and `pytest -q`; every test in the
   repo passes, including the four tinygrad tests once tinygrad is installed
   from git in the venv: `.venv/bin/pip install --no-deps "tinygrad @ git+https://github.com/tinygrad/tinygrad@e837e367aac9"`.
   Then the manual check at the end of `11-python-control-channel.md` with
   the real venv.
2. **E (packaging) with the placeholder app.** `make -C macos python app smoke`.
   Confirm `--list-backends` prints `ort` and `tinygrad` from the embedded
   runtime under the clean environment. Run `macos/scripts/smoke-control.py`
   against the embedded interpreter (from F, or write it now if F is late).
3. **C (Swift core).** Drop the sources in, `make -C macos project test`.
   Fix Swift 6 concurrency errors at the source, not with unsafe annotations.
4. **D (views).** Add, build, run from Xcode with `JETLINK_PYTHON` pointing at
   the venv (`make dev`). Walk every screen against the checklist below.
5. **F (CI).** Push a branch; the CI must be green before merging to main.
6. **G (docs).** Read them once against the running app; fix any string that
   does not match what the app shows.

## Acceptance checklist (run on this Mac, then on a clean user account)

Use a fresh cache folder for the run (Settings > Cache folder > a new empty
directory) so the CoreML compile is exercised once; use tinygrad for the
quick loops.

**Launch and server**
- [ ] `make app` builds `macos/build/Jetlink.app`; `open` it; the Status view
      shows Starting… then Serving within 15 s (ORT probe included).
- [ ] Menu bar extra appears; closing the window leaves it; "Open Jetlink" reopens.
- [ ] Quit stops the server: `pgrep -fl jetlink.server.main` and
      `pgrep -fl jetlink-ort` both empty within 15 s.
- [ ] Kill the app with `kill -9`: the Python server exits within 2 s
      (parent watch) and the ORT worker with it.
- [ ] Kill the server with `kill -9 <pid>`: the app shows Failed, restarts
      after 5 s, shows Serving again; the Logs view has both runs.
- [ ] Sleep assertion: `pmset -g assertions | grep Jetlink` shows the
      assertion while serving on AC; not on battery (unplug if a laptop).

**Models**
- [ ] The list shows 13 models newest first, sizes filled after a refresh,
      the Default tag on BMRLNAP Model v4.
- [ ] Download the default model: progress, rate, then Downloaded; the file is
      `cache/models/a086d5249fc308bb.onnx` and its SHA-256 matches
      (`shasum -a 256`). Cancel a second download mid-way: no `.part` left.
- [ ] Prepare with tinygrad selected: Preparing with the parse/build/save
      stages, then Loaded within about a minute. Inventory shows the `.pkl`
      with `current` true; Status shows Ready.
- [ ] Switch the backend to Automatic, restart the server: the app shows
      Loading for the CoreML session (about 9 minutes on an M1 Pro), and the
      progress message names the elapsed minutes. Then Loaded. Now both
      artifacts appear under "Prepared for".
- [ ] Unload; Load again (artifact exists → the 9 minute load again; verify
      the state is Loading, not Preparing).
- [ ] Delete download keeps the engine (Status remains Ready; the model file is
      gone; "Prepared" stays). Delete prepared engines while loaded: confirmation
      says it unloads; afterwards Status shows None and the files are gone.
- [ ] Add ONNX… with a copy of the same file under another name: Local tag,
      status Downloaded (it is the same sha; the row merges into the catalog
      row? No: the catalog row wins by sha, and `local-models.json` just adds a
      name. Confirm exactly one row for that sha.)
- [ ] Point the cache folder at the old `models_cache/` beside the checkout:
      the existing `e8d821733be15ebe` CoreML artifact appears as an orphan
      ("Unknown model e8d821733be15ebe") unless its sha is resolved by the
      catalog (it is Cinque Terre; the pointer resolves it once the catalog
      loads, so it should show under that name). Delete works on it.

**Comma on the bench** (the real acceptance)
- [ ] With the default model Loaded (CoreML), plug the comma into a USB-A
      port. Status: Connected over USB within 10 s; the comma's icon goes
      green without a pulse phase longer than the handshake; Frames climb at
      about 20 per second; frame time mean under 45 ms; Slow frames 0 over a
      minute. The "Comma" tag appears on that model's row.
- [ ] Unplug: Disconnected with the LinkError text, then Waiting. Replug:
      Connected again with no engine reload.
- [ ] Change the comma's Big Model to a model not prepared here, toggle the
      link off and on: the app shows the comma's request arriving as an
      upload (`stage: upload`), then building; the row for that model gets
      the Comma tag and ends Loaded.
- [ ] Prepare a different model while connected: the confirmation appears;
      accept; the comma falls back and later reconnects to the new model.

**Logs and errors**
- [ ] Logs view scrolls, filters, copies; the file
      `~/Library/Logs/Jetlink/server.log` exists and rotates at 20 MB (test
      by lowering the threshold in a debug build or trust the unit test).
- [ ] Set the Python override to a nonexistent path: Failed with the
      override-missing message; clear it: recovers.
- [ ] Set the cache folder to a read-only directory: server fails to start;
      the last log lines appear in the Status view.

**CI and release**
- [ ] `ci.yml` green on main.
- [ ] Tag `v0.3.0-rc1` on a branch, push: the Release workflow produces the
      unsigned zip (no secrets yet), the sdist and wheel, and either the GHCR
      image or a note that it failed. Delete the pre-release afterwards if it
      was only a test.

## What to record for the user (final report)

- The app's size, launch-to-serving time, download rate seen, tinygrad
  prepare time, CoreML prepare and load times on this Mac.
- The bench numbers from the comma run (fps, mean/p99 frame time, slow frames).
- Every deviation from the plans, by file.
- The secrets that still have to be created for signed releases (none exist:
  `security find-identity -v -p codesigning` shows 0 identities on this Mac).
- The open decision on `pyproject.toml`'s `tinygrad` extra (a git pin cannot
  be published to PyPI; the extra can stay as documentation while
  `requirements-git.txt` and `run-mac.sh` carry the pin).
