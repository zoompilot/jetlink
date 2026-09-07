# Upstreaming plan: two PRs to sunnypilot

Written 2026-09-06 against `sunnypilot-jetson-trt` at `2409aa22bb` (parent
zoompilot baseline `a065958cf2`, sunnypilot master `6135084c94`, commaai
master `9d1f0d4171`) and this repo at `2d6d85b`. Supersedes
`upstreaming-options.md` and `upstreaming-options-codex-2026-09-06.md`; the
design decisions here were made after both, and every finding below was
re-read from the code on the day. Nothing has been changed yet.

Two PRs against `sunnypilot/master`:

1. **PR 1, bug fixes.** selfdrived's "Big Model Ready" chime and the localizer
   alerts on never-received messages. Optionally the model-fetcher manifest
   write. No accelerator code, no schema, no params.
2. **PR 2, accelerators and jetlink.** The whole integration in one PR, in the
   shape a maintainer can review as "chestnut untouched, jetlink added". We
   may split it later; the point now is to show the complete upstreamable
   implementation.

## 1. Decisions

Each of these was argued in the two option documents. This is what we build.

| Axis | Decision | Why |
|---|---|---|
| Chestnut | **Native, at master's locations.** `ChestnutState` stays in `modeld.py`; `modeld_v2` import reverts; no `ChestnutAccelerator`; `chestnut_present()`/`chestnut_compiled()`/the PCIe wait stay in `modeld.main()` verbatim. | The fork's `chestnut.py` carries a copy of the #38742 wait plus a fallback `chestnut_ready` that keeps the wait alive after sunnypilot syncs the revert (#38760). That is fork-owned native policy. A2's "import lazily" still leaves the wait and the HCQ env var in our file. |
| `sunnypilot/accelerators` | **A module API, not a registry.** Eleven functions (section 4.1), one implementation (jetlink). No `Accelerator` Protocol, no `_BACKENDS`, no `_ask`, no `active()`/`catalog()`. | With one implementation the registry answers two questions two ways: `active()` is first *ready*, `catalog()` is first *present*. A fitted-but-uncompiled chestnut plus a cached jetlink gives a chestnut catalog and a jetlink modeld. Selection becomes `if chestnut_present(): native elif accelerators.ready(): jetlink`, and the ambiguity is gone by construction. |
| Opt-in | **`JetlinkEnabled == True` is the only enable.** Drop the "absent means auto via `link_configured()`" rule. `setup.sh` reads the param before touching the gadget or sysctls. | Confirmed: on AGNOS with the package installed, `setup_gadget.sh` creates `/dev/ffs-jetlink/ep0`, `link_configured()` turns true, `enabled()` turns true, and `uses_stock_runner()` silently routes manager away from any custom small bundle. Installation alone enables the feature. |
| Sysctls | **Applied by jetlinkd when enabled, prior values recorded and restored on stop.** | Today: three system-wide `sysctl -w` on every boot, no param gate, no restore path. Also the shipped values (16M/8M/128M) differ from `drive-2026-09-05-ba-latency.md` (32M/8M/64M); reconcile which set the "zero lagging frames over 20 min" number belongs to before shipping either. |
| Wire | **`chestnutPresent` and `chestnutState` mean comma's board only.** jetlink stops widening `usb.py` and stops publishing `chestnutState` with a synthetic `pcieLtssm = 0x78`. Runtime status goes in additive `ModelDataV2SP` fields; offroad progress stays in `AcceleratorProgress`; telemetry is deferred until a `customReserved` slot is assigned by maintainers. | Removes the `accel.name == "chestnut"` filter in hardwared and every `catalog()` compensation in the model manager. `validate_sp_cereal_upstream.py` passes at HEAD and only checks union discriminants, so additive SP fields are safe; a new top-level Event variant is not ours to allocate. |
| selfdrived | **Native block verbatim plus one call into a sunnypilot adapter.** jetlink stops writing `ChestnutLoading`/`ChestnutActive`. | The 5 s settling window is restored for chestnut. The adapter reads `modelV2.big` and `modelDataV2SP.bigModelAvailable`, which is what the joining state already exposes. |
| UI | **Master's `_update_chestnut_state` verbatim; a jetlink view in `UIStateSP` consulted only when no board is present.** | HEAD's rewrite changes four observable things for a chestnut user (unplug shows DISCONNECTED not FAILED, hot-plug lands in FAILED, LOADING before UNCOMPILED at ignition, readiness recomputed at 5 Hz with a sysfs glob). |
| Packaging | **Submodule `jetlink_repo`, symlink line in `launch_chffrplus.sh`, `release.json` and `verify_release.py` gone from the fork.** | The submodule sha is the pin. The Jetson image records its own ID (`docs/releasing.md`). |
| modeld hooks | **An `elif JETLINK:` path, five sites, chestnut block byte-identical to master.** No loader-thread restructure, no `abandoned` lock, no frame-age filter. | The joining state returns immediately with the small model driving and joins in the background, so it needs no 60 s loader thread. The lock and the diagnostics were what pushed the hunk from 5 sites to 194 lines. |

Things that stay exactly as they are: the joining state, the fallback reset,
the transport, the warp scons target, jetlinkd's provisioning and dormancy,
the shutdown handshake, `models.json`, the engagement gate for promotion.
Section 3 lists them as acceptance criteria.

## 2. PR 1: bug fixes

Branch `sp/big-model-fixes` from `sunnypilot/master`. Three commits; the third
is droppable if a reviewer wants the PR narrower.

### 2.1 selfdrived: chime on a real big frame

Confirmed on master `selfdrived.py:198-201`: the `bigModelReady` chime fires
on any True-to-False edge of `ChestnutLoading`. modeld master writes
`ChestnutActive=False` at line 309 on a failed or timed-out load and
`ChestnutLoading=False` at line 316 a few seconds later, after the small
model constructs. So a failed load produces "Big Model Failed" followed by a
"Big Model Ready" chime, with the small model driving. The fix chimes on
`modelV2.big` rising while alive and valid, which is the only signal that a
big frame was published. `warmup_sec`, `big_model_settling` and every other
line of the block stay.

```python
# __init__
self.big_model_running = False
# update_events, after the loading block
running_big = self.sm.alive['modelV2'] and self.sm.valid['modelV2'] and self.sm['modelV2'].big
if running_big and not self.big_model_running:
  self.events_sp.add(custom.OnroadEventSP.EventName.bigModelReady)
self.big_model_running = running_big
```

Test: a `selfdrived` unit test that feeds the (b) trace from the audit
(`ChestnutLoading` True, `ChestnutActive` False, `ChestnutLoading` False, no
`big` frame) and asserts no `bigModelReady`; and the (a) trace with a `big`
frame and asserts exactly one.

### 2.2 selfdrived: no localizer alerts from a message never received

Master `selfdrived.py:456-460` reads `deviceMotion.posenetOK`, `inputsOK` and
`vehicleParameters.valid` unconditionally. A SubMaster message never received
is the capnp default, so all three read False. locationd is `poll='cameraOdometry'`
and publishes nothing until modeld does. On a non-chestnut device that has
passed the 6 s `initialized` window with no `deviceMotion` yet, `posenetInvalid`
and `locationdTemporaryError` (both NO_ENTRY and SOFT_DISABLE) fire on
defaults. `processNotRunning` already covers a dead locationd. The fix gates
each on `self.sm.seen[...]`. This is parent-branch work (commits `d0115f6a68`,
`44d75a6682`); we own the PR, not the attribution problem.

### 2.3 fetcher: write a chunk manifest only for chunks on disk (optional)

Master `ModelParser._parse_artifact` writes `<fileName>.chunkmanifest` for
every chunked artifact in both catalogs on every 1 Hz manager tick and every
UI parse, whether or not the artifact was downloaded. On a fresh model root
that is 89 stray files today (77 qcom, 12 chestnut). The fork's
`_repair_chunk_manifest` writes only when chunk 0 exists and the count differs.
**Reframe the commit message**: the original claim (same file name in both
catalogs with different chunk counts thrashing at 2 Hz) is not reproducible on
today's live manifests, so lead with the stray-manifest fact, which is. Tests
`TestChunkManifestRepair` come with it.

Not in PR 1: the `ChestnutModelError` key. commaai removed it in #38760;
sunnypilot still has it. That is a sync decision for sunnypilot.

## 3. Jetlink behaviour to preserve (PR 2 acceptance criteria)

Read out of `joining.py`, `backend.py`, `model_state.py`, `fallback.py` and
`jetlinkd.py` at HEAD. Any refactor that changes one of these is a regression.

| Situation | Behaviour, with the code that implements it |
|---|---|
| Provisioned Jetson boots after the comma | `make_model_state` returns at once with the small model driving; `_join_loop` connects in the background (`backend.py:171-229`, `joining.py:105-183`). |
| Promotion | Only with fresh (`ENGAGEMENT_MAX_AGE = 0.25` s) valid `selfdriveState`, `carState`, `carControl`, and `enabled`, `latActive`, `longActive` all false (`joining.py:279-282, 450-453`). Standstill is tracked but not a condition. |
| First big frame | `ChestnutActive`-equivalent readiness and progress cleared only after the first successful inference (`joining.py:246-254`). Availability is signalled separately (`bigModelAvailable`). |
| Failure after promotion | Demote to small, reset small's queues in place via the captured JIT (`fallback.py:10-33`), re-run the same frame on small (`joining.py:233-245`), back off `min(5 * 2**(n-1), 60)` s with the counter reset after a 60 s stable join (`joining.py:332-347`). |
| Runtime deadline | `INFERENCE_TIMEOUT = 0.5` s per frame once joined (`backend.py:35, 273`); 3 s only for hello/ensure. |
| Ready link waiting for a window | Keepalive ping every 10 s, failure schedules a rejoin (`joining.py:398-438`). |
| Realtime inheritance | `warp_cache.init_device()` runs before `config_realtime_process`; every jetlink thread drops realtime first; the FunctionFS reader re-pins to FIFO 51 off core 7 (`backend.py:57-101`, `joining.py:76-99`, `ffs.py:435-490`). |
| Endpoint ownership | jetlinkd offroad, modeld onroad, manager's `only_offroad` gate; `_close_retired` only on the join loop so unbind never blocks a frame (`joining.py:323-330`). |
| Provisioning | Registry identity, no hashing on a parked car, `JetlinkEngineReady`/`JetlinkSpec` survive reboot, re-asked once per attach (`jetlinkd.py`). |
| Dormancy and wake | `DORMANT_HOLD = 60` s from process start, `/dev/shm/jetlink-dormant` written before the link closes so presence never blinks, CC pin for presence while dormant (`jetlinkd.py:334-335`, `helpers.py:215-237`). |
| Shutdown | hardwared asks jetlink before `DoShutdown`; 20 s wake plus 5 s request, 25 s bound (`backend.py:316-336`, `jetlinkd.py:355-375`). |
| Warp | scons target, presence-only cache, `prepare()` refuses without it (`accelerators/SConscript`, `warp_cache.py`). |

## 4. PR 2: accelerators and jetlink

Branch `sp/jetlink` from `sunnypilot/master`, after PR 1 merges (or rebased
onto it: the jetlink adapter relies on the chime firing on `modelV2.big`).

### 4.1 The module API

`openpilot/sunnypilot/accelerators/__init__.py` exposes, with jetlink as the
only implementation behind it:

```
present() -> bool                     usb-independent: jetlink's gadget/dormant presence
ready() -> bool                       params-only: enabled, no gadget error, spec == selected == engine ready
unavailable_reason() -> str | None    only when opted in
prepare() -> bool                     warp cached, tinygrad device init before realtime
make_model_state(cam_w, cam_h, small) the joining state (same duck type as today)
make_status_publisher(pm, model)      replaces make_health_publisher; see 4.6
progress() / report_progress() / clear_progress()
shutdown(reason, timeout)             bounded in the core, see 4.8
uses_stock_runner() -> bool           explicit selection only, see 4.7
model_choices() / select_model(name) / active_model_name()
daemons() -> list[Daemon]             jetlinkd, offroad, gated on enabled()
```

Gone: `base.py`'s `Accelerator` Protocol, `_BACKENDS`, `backends()`, `_ask()`,
`active()`, `catalog()`, `name`, `chestnut.py`. Keep `Daemon` as a two-field
description in `__init__.py` (process_config still needs one). A package
missing at import time makes every function answer its negative default;
that replaces the swallowed `ImportError` of the registry.

### 4.2 modeld.py: five sites, chestnut verbatim

Against master `6135084c94` line numbers. The `if CHESTNUT:` block, the
poller wait, `ChestnutState`, `HCQDEV_WAIT_TIMEOUT_MS` and the fallback
`except` stay byte-identical.

1. **Select, between 263 and 265 (before `config_realtime_process`):**
   ```python
   JETLINK = not CHESTNUT and accelerators.ready() and accelerators.prepare()
   ```
   `prepare()` must run here because `warp_cache.init_device()` spawns
   tinygrad's device thread, which would otherwise inherit FIFO 54 on core 7.
2. **Load, after the `if CHESTNUT:` block (311) and before 313:**
   ```python
   elif JETLINK:
     small_model = ModelState(vipc_client_main.width, vipc_client_main.height, False)
     try:
       model = accelerators.make_model_state(vipc_client_main.width, vipc_client_main.height, small_model)
     except Exception:
       cloudlog.exception("jetlink load failed")
       model = None
   ```
   and line 313 becomes `if small_model is None: small_model = ...` with its
   original condition. No loader thread; the joining state returns at once.
   Line 316 `params.put_bool("ChestnutLoading", False)` stays as is: jetlink
   no longer writes that key (4.5), and on the jetlink path it is already
   False from 259.
3. **Publisher, 321 and 327:** `pub_socks` gains the jetlink status service
   only if 4.6 lands a service; with SP fields only, nothing. Line 327 stays;
   add `if JETLINK: chestnut_state = accelerators.make_status_publisher(pm, model)`
   so the `after_enqueue` call at 439 carries the same signature.
4. **Fallback, before 441:** `if JETLINK: raise`. The joining state owns its
   own demotion; a small-model exception is fatal as on stock. Without this a
   small-model fault while the Jetson is active would make master's handler
   write `ChestnutActive=False` and orphan the joining state's threads and
   open link.
5. **SP field, near 471:**
   `mdv2sp_send.modelDataV2SP.bigModelAvailable = getattr(model, 'big_model_available', False)`
   plus the status fields from 4.6.

`modelV2.big = model.chestnut` at 466 is untouched; `JoiningModelState.chestnut`
already proxies it. The `loading` duck-type conditionals, `_boottime_ns`, the
frame-age filter and its log, and the `check_modeld_pkl` parent hunk are not
in this PR.

### 4.3 The joining state loses its param writes

`joining.py:177` (remove `ChestnutActive`), `:271` and `:276` (put
`ChestnutLoading`/`ChestnutActive` at init, swap and demote) go. Its state
travels in `modelV2.big` (proxied `chestnut`) and `big_model_available`, and
in a new `big_model_state` property (4.6). `report_progress`/`clear_progress`
calls stay (the UI reads them offroad and during the first join).

### 4.4 selfdrived: one call

Keep master lines 198-218, 401-402 and 434-462 verbatim, including
`warmup_sec`. Add `openpilot/sunnypilot/selfdrive/selfdrived/accelerator_events.py`
with a `AcceleratorEvents.update(sm, enabled, events, events_sp)` that:

- adds `bigModelAvailable` on `modelDataV2SP.bigModelAvailable` rising while
  `not modelV2.big` (moved from HEAD `:189-197`);
- adds `bigModelLoading` NO_ENTRY while `modelDataV2SP.acceleratorState == joining` and `not alive['modelV2']` (a late join never blocks);
- adds a sunnypilot `bigModelLinkLost` SOFT_DISABLE plus PERMANENT with the
  "Small model is driving, reconnecting if it comes back" text on
  `modelV2.big` falling while `enabled`, and nothing on a fall while
  disengaged.

selfdrived calls it once after the native block. Native `bigModelFailed` and
its "Restart the car to retry" text are untouched and never fire for jetlink,
because jetlink never writes `ChestnutActive`. The `bigModelReady` chime is
backend-neutral after PR 1.

### 4.5 hardwared and usb.py

`usb.py`: master verbatim. `hardwared.py`: master's `chestnut_valid` line
verbatim (the name filter goes with `chestnutState` no longer being
jetlink's), plus two direct calls that stay from HEAD:

```python
accelerator_error = accelerators.unavailable_reason()
set_offroad_alert_if_changed("Offroad_AcceleratorUnavailable", accelerator_error is not None, extra_text=accelerator_error)
...
accelerators.shutdown(f"comma shutting down, offroad since {off_ts}", timeout=25.0)
```

`shutdown` enforces the bound in the core with a thread and a join, so no
backend can hold hardwared longer than stated; a comment says deviceState
pauses for that long. On a device with jetlink disabled it costs one param
read and returns.

### 4.6 Status and schema

Additive `ModelDataV2SP` fields, all default-zero, ordinals after the last
in use at master:

```
bigModelAvailable @3 :Bool;            # already in the fork
acceleratorState  @4 :AcceleratorState;  # enum: none, joining, running, retrying, unavailable
acceleratorName   @5 :Text;            # "jetlink"
```

Offroad progress stays in `AcceleratorProgress`. Telemetry (temp, power, GPU
load, clock, supply) is **not** in this PR: it needs a `customReserved` slot
assigned by maintainers, and until then it goes to swaglog at 1 Hz from the
status publisher. `make_status_publisher(pm, model)` returns an object whose
`send()` has `JetlinkHealth.send`'s signature (it is the `after_enqueue`
callback, needed for the `want_state` piggyback) but publishes nothing on
`chestnutState`. Run `validate_sp_cereal_upstream.py` in CI as the fork
already does.

### 4.7 Model manager and runner selection

- `uses_stock_runner()` becomes `JetlinkEnabled is True and JetlinkModel is set`.
  Not `link_configured()`, not `ready()`: configuration, so a late boot cannot
  change which modeld manager runs mid-drive, but explicit configuration.
- `get_active_bundle`'s short-circuit and the `ModelRunnerTypeCache` clears in
  `select_model`/the link toggle stay. `manager.py:329-332` (skip
  `maybe_apply_default_model` under the override) stays.
- Every `accelerators.catalog() == "chestnut"` reverts to master's
  `chestnut_present()` / `deviceState.chestnutPresent`: `models/helpers.py:131-132`,
  `fetcher.py:193-194`, `manager.py:39, 284, 325-328, 347`, `default_model.py:15`,
  `model_info.py:15, 75`, `sunnypilot/ui_state.py:48, 158-160`, and their
  test renames.
- **Fallback label fix**: one helper, `effective_small_bundle()`, returns None
  when `uses_stock_runner()` and the stored qcom bundle otherwise; used at
  `model_info.py:80` and `:104` and at `sunnypilot/ui_state.py:163`, which
  also stops `:167` forcing `chestnut_compiled` true off a bundle that is not
  running.

### 4.8 UI

- `ui_state.py`: master's `_update_chestnut_state` and the latched
  `chestnut_compiled()` verbatim. Add at the top:
  `if self.accelerator_view is not None: self.chestnut_state = self._accelerator_state(); return`.
  `_accelerator_state` is HEAD's rule set (progress stage offroad, `running_big`
  first onroad, DISCONNECTED on `not present`, LOADING while joining).
- `sunnypilot/ui_state.py`: build `accelerator_view` in the 5 Hz params pass
  from `accelerators.present()`, `ready()`, `progress()`, `uses_stock_runner()`
  and `modelDataV2SP.acceleratorState`, and only when
  `not self.sm['deviceState'].chestnutPresent`.
- `usb_unknown` guard: `not (accelerators.present() or any(is_chestnut_usb_id(...)))`.
  Comma's #38745 needs it regardless because the comma is the gadget.
- mici models panel: keep the link toggle and the accelerator picker as they
  are. Non-mici `layouts/settings/models.py`: add the same toggle and picker,
  and fix `_status_note` (`:207-228`) so it does not say "when the chestnut
  is ready" on a jetlink device. The supported scope is both layouts.

### 4.9 Boot, setup, sysctls, packaging

- `launch_chffrplus.sh`: `ln -sfn jetlink_repo/jetlink jetlink` next to the
  tinygrad line; the `./openpilot/sunnypilot/accelerators/setup.sh` call stays.
- `accelerators/setup.sh` loops as today; `jetlink/setup.sh` starts with a
  `python3 -c` read of `JetlinkEnabled` and exits 0 unless it is true. The
  candidate-path loop, the `ln -sfn`, `release.json` and the
  `verify_release.py` call go.
- Sysctls move to `jetlinkd.run()`: read current values into
  `/dev/shm/jetlink-sysctl-prev`, apply, and restore in the exit path and in
  the "disabled, releasing the link" branch. SIGKILL leaves them until reboot;
  say so in the comment. Reconcile the two value sets first (section 1).
- `.gitmodules`: add `jetlink` at path `jetlink_repo`; the opendbc URL is
  sunnypilot's (the zoompilot pointer is parent-branch and must not appear).
- `accelerators/SConscript`: candidate list collapses to `#jetlink_repo`.
- `params_keys.h`: the eight keys, including `JetlinkCachedModels` which
  CLAUDE.md's table omits. `process_config.py`: the `daemons()` spread stays.
- `helpers.enabled()` is `bool(_get(P_ENABLED))`; `opted_in()` collapses into it.

### 4.10 Tests and docs

- Keep `accelerators/conftest.py` (master has no conftest, but `OpenpilotPrefix`
  for the subtree is the safe default). Fix the two tests that reach live
  params if it were absent: `test_jetlinkd.py:396-403` (`set_engine_ready`)
  and `test_joining.py` (`report_progress`), by patching.
- Delete `test_accelerators.py`'s registry tests; add selection tests for the
  module API (disabled, enabled-not-ready, ready; `uses_stock_runner` only
  with model set).
- New: native-equivalence tests that run modeld's decide/load block with
  `chestnut_present()` patched True and assert `accelerators.*` is never
  called; selfdrived traces (a) and (b) against the adapter; `carrying_model`
  with a stored non-default qcom bundle under the override.
- Stale text to fix while there: `joining.py:257-266` warmup docstring (the
  small model is not warmed by modeld on master or HEAD), `model_state.py:163-165`
  and `docs/openpilot-integration.md` still saying 3 s where the frame deadline
  is 0.5 s, `base.py:85-88` "borrow its warp", `ffs.py:438-440, 468-469`
  reader "created from modeld's frame thread", CLAUDE.md's params table
  (`JetlinkCachedModels`) and `libusb_event` creator (answered by
  `warp_cache.py:82-84`).

### 4.11 Expected upstream-owned footprint

| File | Lines (estimate) | Content |
|---|---|---|
| `selfdrive/modeld/modeld.py` | ~20 | the five sites in 4.2 |
| `selfdrive/selfdrived/selfdrived.py` | ~3 | import and one call |
| `selfdrive/ui/ui_state.py` | ~6 | view short-circuit and `usb_unknown` |
| `system/hardware/hardwared.py` | ~6 | alert and bounded shutdown |
| `common/hardware/usb.py` | 0 | |
| `selfdrive/selfdrived/events.py` | 0 | |
| `system/manager/process_config.py` | 4 | |
| `launch_chffrplus.sh` | 3 | |
| `common/params_keys.h` | 12 | |
| `selfdrive/selfdrived/alerts_offroad.json` | 4 | |
| `cereal/custom.capnp` | ~8 | additive |

Count it, do not guess it, before the PR:

```bash
git diff --numstat sunnypilot/master..HEAD -- . ':!openpilot/sunnypilot' ':!*/tests/*' ':!openpilot/selfdrive/ui/sunnypilot' ':!openpilot/selfdrive/ui/mici'
```

## 5. Work packages

Each is a subagent task with a branch, a goal, inputs it may read, and a
done-check. W0 first; W1 and W2 form PR 1; W3 before W4 to W9, which can run
in parallel on the same branch if they stay in their files; W10 last.

| # | Package | Goal | Files | Done when |
|---|---|---|---|---|
| W0 | Branches | `sp/big-model-fixes` and `sp/jetlink` cut from `origin/master` (sunnypilot) in fresh worktrees; `.gitmodules` opendbc URL verified as sunnypilot's; venv builds `libparams_c` (see `mac-fork-build-and-tests` memory). | worktrees only | `git merge-base` is master; `scons -j8 openpilot/common/params_pyx.so` succeeds. |
| W1 | PR 1 selfdrived | 2.1 and 2.2 with tests. | `selfdrived.py`, its tests | Tests for traces (a) and (b) pass; `git diff --numstat` shows only selfdrived and tests; settling window untouched. |
| W2 | PR 1 fetcher | Cherry-pick `6108197cac`, reword the message per 2.3, keep tests. | `models/fetcher.py`, `test_manager_download.py` | `TestChunkManifestRepair` passes on the clean branch. |
| W3 | Module API | 4.1 and 4.3: rewrite `accelerators/__init__.py`, delete `base.py`/`chestnut.py`, port `jetlink/` (backend, joining, helpers, jetlinkd, state) to the module API, remove the param writes from joining, narrow `enabled()`, add `big_model_state`. | `sunnypilot/accelerators/**` | Package tests pass with no hardware; grep finds no `ChestnutActive`/`ChestnutLoading` write under `accelerators/`; no `chestnut_ready` anywhere under it. |
| W4 | modeld | 4.2 against master's file. | `selfdrive/modeld/modeld.py` | `git diff origin/master -- modeld.py` shows only the five hunks; `ChestnutState` present at its master lines; `ContractTest` in `test_joining.py` passes. |
| W5 | selfdrived adapter and schema | 4.4 and 4.6: adapter, SP events, capnp fields, modeld's SP writes. | `sunnypilot/selfdrive/selfdrived/accelerator_events.py`, `custom.capnp`, SP events, modeld site 5 | `validate_sp_cereal_upstream.py` passes; selfdrived diff is import plus one call; native `events.py` unchanged. |
| W6 | Model manager | 4.7 including `effective_small_bundle`. | `sunnypilot/models/*`, `model_info.py`, `sunnypilot/ui_state.py`, tests | `test_manager_download.py` passes; new test: stored non-default qcom bundle plus override reports the default small name; `catalog` string absent from the tree. |
| W7 | UI | 4.8 for `ui_state.py`, `UIStateSP`, mici and non-mici layouts. | `selfdrive/ui/**` | `git diff origin/master -- selfdrive/ui/ui_state.py` shows only the short-circuit and `usb_unknown`; `test_mici_settings.py` passes; non-mici layout has toggle and picker. |
| W8 | Boot and packaging | 4.9 and 4.5: setup gating, sysctls in jetlinkd with restore, hardwared, submodule, SConscript, params, process_config. | `launch_chffrplus.sh`, `setup.sh`s, `jetlinkd.py`, `hardwared.py`, `.gitmodules`, `SConscript`, `params_keys.h` | With `JetlinkEnabled` absent, `setup.sh` exits before `setup_gadget.sh`; jetlinkd writes and restores the sysctl record in a unit test with `sysctl` mocked; hardwared diff is the two calls. |
| W9 | Docs | 4.10 stale text, `docs/openpilot-integration.md` rewritten for the module API, CLAUDE.md corrections, PR descriptions with the behaviour matrix from section 3. | docs, CLAUDE.md, docstrings | No doc or comment says 3 s frame deadline, "borrow its warp", or "warmed on the main thread". |
| W10 | Validation harness | Native-equivalence tests (4.10) and a merge-replay script: `git apply --check` of the #38760 revert and the #38684 fused-warp change against `sp/jetlink`. | `accelerators/tests/`, `scripts/` | Both applies check clean or the failing hunks are only in `sunnypilot/`. |

Subagent rules: read the audits in this document's section references before
editing; never touch a chestnut line in an upstream-owned file; run `ruff check`
and the package tests before reporting; report with `file:line` and the
`git diff --numstat` of upstream-owned files.

## 6. Verification I can complete myself

In order, on the Mac unless marked. Each has a pass condition.

1. **Footprint.** `git diff --numstat origin/master..sp/jetlink` filtered as in
   4.11. Pass: every upstream-owned file within twice the estimate, and
   `git diff origin/master -- openpilot/selfdrive/modeld/modeld.py` contains
   no hunk touching a line inside `class ChestnutState`, the poller wait, the
   `if CHESTNUT:` load block, or the `except Exception` fallback.
2. **No backend names upstream.** `grep -rn jetlink openpilot/selfdrive openpilot/system openpilot/common` returns only `params_keys.h` and the `accelerators` import lines. `grep -rn 'accelerators\.' openpilot/selfdrive/modeld/modeld.py` returns exactly the five sites.
3. **Ruff.** `ruff check` clean on both branches, remembering this repo allows
   implicit string concatenation and the fork does not.
4. **Unit suites.** From the fork root with the venv (per the
   `mac-fork-build-and-tests` memory, venv bin first on PATH, `libparams_c`
   rebuilt): `pytest openpilot/sunnypilot/accelerators openpilot/sunnypilot/models/tests openpilot/sunnypilot/modeld_v2/tests openpilot/selfdrive/selfdrived/tests`. Pass: all green, and `modeld_v2/tests` imports `ChestnutState` from `selfdrive/modeld/modeld.py`.
5. **Cereal.** `validate_sp_cereal_upstream.py -g` on `sp/jetlink`, then `-r`
   against a `git archive upstream/master` cereal with upstream's pinned
   opendbc `car.capnp`, mirroring `.github/workflows/cereal_validation.yaml`.
   Pass: "cereal compat OK".
6. **Merge replay.** `git apply --check` of the #38760 revert
   (`e3def37695`, modeld/helpers/params hunks) and the #38684 fused-warp
   change against `sp/jetlink` in a disposable worktree. Pass: clean, or
   conflicts confined to `openpilot/sunnypilot/`.
7. **Native-equivalence trace.** The W10 tests: with `chestnut_present()`
   patched True, `accelerators.ready/prepare/make_model_state` are never
   called; with it False and `JetlinkEnabled` absent, likewise; selfdrived
   traces (a) and (b) produce the master event sequence plus PR 1's fix.
8. **Opt-in is inert.** In a container or on the comma over ssh: with
   `JetlinkEnabled` absent and the package installed, `setup.sh` exits before
   `setup_gadget.sh` (no `/dev/ffs-jetlink`), `sysctl vm.dirty_bytes` is the
   stock value, `get_active_model_runner()` follows the stored bundle, and
   `accelerators.present()/ready()` are False.
9. **Fallback label.** Unit test from W6, and on the bench: select a custom
   small bundle, enable the link, kill the server mid-drive on the live
   bench, read the models panel. Pass: it names the default small model.
10. **Bench, hardware present.** `tools/jetlink_live_bench.sh 180` on the
    comma with the Jetson attached, per CLAUDE.md, after stopping jetlinkd.
    Pass: one join, `modelV2.big` true, zero link failures, and on ignition
    off the jetlinkd log shows the sysctl restore line. Then a parked cycle:
    jetlinkd goes dormant at 60 s, Jetson suspends, USB edge wakes it.
11. **Shutdown bound.** With the Jetson unplugged and `JetlinkEnabled` true,
    trigger hardwared's shutdown path on the bench and time
    `accelerators.shutdown`. Pass: returns within 25 s and `DoShutdown` is
    written after it. With `JetlinkEnabled` absent: returns in under 10 ms.

Items 8 to 11 need the bench Jetson and comma from the `bench-hosts` memory
and are run over ssh; if the bench is down they are reported as not run, not
as passed.
