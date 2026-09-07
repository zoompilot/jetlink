# PR 1 to sunnypilot: big model fixes

Branch `sp/big-model-fixes` on `sunnypilot/master` (`6135084c94`). Three commits, the
middle one droppable if a reviewer wants the PR narrower.

## Title

selfdrived: chime Big Model Ready on a real big frame, gate localizer alerts on seen

## Description

Three small fixes around the big model and the localizer, each with a unit test.

**selfdrived: gate the localizer alerts on SubMaster.seen.** A message never received
reads as the capnp default, so `deviceMotion.posenetOK`, `inputsOK` and
`vehicleParameters.valid` are all False before locationd has published anything.
locationd polls `cameraOdometry` and publishes nothing until modeld does. On a device
that passes the 6 s initialized window with no `deviceMotion` yet, `posenetInvalid` and
`locationdTemporaryError` (both NO_ENTRY and SOFT_DISABLE) fire on defaults.
`processNotRunning` already covers a dead locationd. Each check is now gated on
`sm.seen[...]`. Test: `test_localizer_alerts.py`, shown to fail against master's file.

**models: write the chunk manifest for the chunks that are on disk.**
`ModelParser._parse_artifact` wrote `<fileName>.chunkmanifest` for every chunked artifact
in both catalogs on every 1 Hz manager tick and every UI parse, downloaded or not: 89
stray files on a fresh model root (77 qcom, 12 chestnut). The manifest is now written only
when chunk 0 exists and the recorded count differs. Test: `TestChunkManifestRepair`.

**selfdrived: chime Big Model Ready on the first big frame.** The `bigModelReady` chime
fired on any True to False edge of `ChestnutLoading`. modeld writes `ChestnutActive=False`
on a failed or timed out load and clears `ChestnutLoading` a few seconds later once the
small model is up, so a failed load played "Big Model Failed" and then "Big Model Ready"
with the small model driving. The chime now fires on `modelV2.big` rising while alive and
valid, the only sign a big frame was published. The falling edge still starts the 5 s
settling window; `warmup_sec`, `big_model_settling` and the `bigModelFailed` logic are
unchanged. Test: `test_big_model_ready.py` (failed load: no chime; good load: exactly one;
big while not alive: none; re-arms after a fall).

## Verification

- `pytest openpilot/selfdrive/selfdrived/tests openpilot/sunnypilot/models/tests/test_manager_download.py`: 72 passed, 1 skipped
- ruff clean on the changed files
- footprint: `selfdrived.py` +9/-3, `fetcher.py` +37/-12, plus tests
