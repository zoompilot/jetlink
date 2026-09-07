# PR 2 to sunnypilot: accelerators and jetlink

Branch `sp/jetlink`, stacked on `sp/big-model-fixes` (the adapter relies on the chime
firing on `modelV2.big`). Seven commits, one per subsystem, reviewable in order.

## Title

accelerators: run the big driving model on an attached Jetson (jetlink)

## Description

Adds `openpilot/sunnypilot/accelerators`, a small module API with one implementation:
jetlink, which runs comma's big driving model on a Jetson attached over USB (the comma is
the FunctionFS gadget, the Jetson the host; TensorRT on the Jetson, the camera warp on the
comma). The design rule throughout is **chestnut untouched, jetlink added**: comma's board
keeps its code at master's locations byte for byte, and jetlink is an `elif` beside it.

What a maintainer should see in the upstream-owned files (11 files, +90/-2):

| File | Change |
|---|---|
| `selfdrive/modeld/modeld.py` | import plus five hunks: `JETLINK = not CHESTNUT and accelerators.ready() and accelerators.prepare()` before `config_realtime_process`; an `elif JETLINK:` load; the status publisher; `if JETLINK: raise` at the top of the fallback handler; three `modelDataV2SP` fields. `ChestnutState`, the poller wait, the `if CHESTNUT:` block and the fallback body are unchanged (a test asserts this against the base). |
| `selfdrive/selfdrived/selfdrived.py` | import, one constructor line, one `AcceleratorEvents.update()` call after the native big model block. |
| `selfdrive/ui/ui_state.py` | a short-circuit at the top of `_update_chestnut_state` when an accelerator view exists, and the `usb_unknown` guard (the comma is the gadget and enumerates nothing). |
| `system/hardware/hardwared.py` | the offroad alert and a bounded `accelerators.shutdown()` before `DoShutdown`. |
| `system/manager/process_config.py` | `*accelerators.daemons()` (jetlinkd, offroad only). |
| `common/params_keys.h`, `cereal/custom.capnp`, `alerts_offroad.json`, `launch_chffrplus.sh`, `.gitmodules`, `.gitignore` | eight params; additive `ModelDataV2SP.bigModelAvailable/acceleratorState/acceleratorName` and `OnroadEventSP.bigModelAvailable/bigModelLinkLost`; one offroad alert; one symlink line; the `jetlink_repo` submodule. |

Everything else lives under `openpilot/sunnypilot/`: the module API and backend, the
selfdrived adapter (`bigModelAvailable`, `bigModelLoading` only while joining with modelV2
dead, `bigModelLinkLost` on a fall while engaged), the model manager override, the UI view
and the models-panel toggle and picker in both layouts, and the tests.

**Opt-in is explicit.** `JetlinkEnabled == True` is the only enable: installing the
submodule enables nothing, `setup.sh` reads the param before presenting the gadget, and
jetlinkd applies the three VM sysctls only when enabled, records the prior values and
restores them on exit.

**Cereal keeps comma's meaning.** `deviceState.chestnutPresent` and `chestnutState` mean
comma's board only; jetlink publishes nothing on them. Runtime state travels in additive
`ModelDataV2SP` fields; Jetson telemetry goes to swaglog until maintainers assign a
`customReserved` slot. `validate_sp_cereal_upstream.py` passes.

**Two review findings fixed before this PR.** A lost link latches a native `bigModelFailed` plus the sunnypilot `bigModelLinkLost` every tick until the driver disengages, because the main state machine only reads native events and cancels a soft disable the tick its event vanishes. jetlinkd restores the VM sysctls only when the link is turned off, not at the ignition handoff where manager stops it; ratio-mode limits are restored through their ratio keys because the kernel silently skips a write of 0.

**Behaviour the tests pin.** The joining state returns at once with the small model
driving and joins in the background; promotion needs fresh, valid, disengaged
`selfdriveState`/`carState`/`carControl`; a failure after promotion demotes in place,
re-runs the frame on the small model and backs off `min(5 * 2**(n-1), 60)` s; the per-frame
deadline is 0.5 s; every jetlink thread drops realtime and the FunctionFS reader re-pins
off core 7; jetlinkd owns the link offroad and modeld onroad.

## Verification

- 501 passed, 3 skipped across accelerators, models, modeld_v2, selfdrived, sunnypilot selfdrived and ui
- ruff clean on every changed file; `modeld`, `hardwared`, `process_config`, `modeld_v2` import on a PC
- `validate_sp_cereal_upstream.py`: "cereal compat OK"
- `merge_replay.sh`: commaai #38760 (chestnut wait revert) adds two conflict regions in `modeld.py` (the `JETLINK =` line and the `elif JETLINK:` load) beyond master's own; #38684 (fused warp) conflicts identically on master and this branch
- On the comma with `JetlinkEnabled` absent: `setup.sh` exits before the gadget, `present()/ready()/uses_stock_runner()` False, `accelerators.shutdown()` returns in under 1 ms

Not exercised in CI: the live bench (camerad plus the real modeld against a Jetson) and
the shutdown handshake with a Jetson that does not answer; both need the hardware.

## Notes for review

- `models/manager.py` is untouched: master has no default-model bootstrap to skip.
- Capnp ordinals: `OnroadEventSP` `bigModelAvailable @26`, `bigModelLinkLost @27`;
  `ModelDataV2SP` `@3`..`@5` plus the `AcceleratorState` enum.
- The submodule points at `zephleggett/jetlink`; the package is MIT.
