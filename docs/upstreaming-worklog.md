# Upstreaming work log

Execution record for `upstreaming-plan.md`. Started 2026-09-06. Each entry is
what was done, where, and what was verified, so the state can be picked up
from this file alone.

## Layout

| Worktree | Branch | Base | Purpose |
|---|---|---|---|
| `sunnypilot_proj/zoompilot-fixes` | `develop-big-model-fixes` | `develop` (`75580ad1e3`, the 2026-09-06 squash of danger-unstable) | PR 1 content for zoompilot, no jetlink |
| `sunnypilot_proj/sunnypilot-jetson-trt` | `jetson-trt` | tag `jetson-trt-pre-upstreaming` = `2409aa22bb` | the full plan, zoompilot-compatible |
| `sunnypilot_proj/sp-big-model-fixes` | `sp/big-model-fixes` | `origin/master` `6135084c94` | sunnypilot PR 1 |
| `sunnypilot_proj/sp-jetlink` | `sp/jetlink` | `origin/master` `6135084c94` | sunnypilot PR 2 |

Fresh worktrees need submodules and two build products; the script that does
it is `scratchpad/setup_wt.sh` (submodule `--reference` must point at the
common git dir's `modules/<name>`, a linked checkout is refused). Tests run as
`PYTHONPATH=<worktree> ../sunnypilot/.venv/bin/python -m pytest -p no:cacheprovider`.

## Decisions made during execution

- **Sysctl values: 128 MB / 16 MB / 8 MB (min_free / dirty / dirty_background).**
  The plan flagged two sets. `f769de75b0` (2026-09-06) raised 64/32 to 128/16
  after the contention pass in the `ba-root-cause` memory: under a 500 MB hog
  plus five CPU workers, 64/32 let one 104 ms frame through and 128/16 did not.
  The drive doc's table at line 217 records the earlier set; the newer one wins.
- **The seen guards are already in `develop`** (`75580ad1e3` carries
  `d0115f6a68` and `44d75a6682`), so the zoompilot branch needs only the chime
  fix and the fetcher fix. The sunnypilot PR 1 needs all three.
- **The fork's `test_big_model_availability.py`** tests the late-join
  `bigModelAvailable` chime, which moves into the PR 2 adapter; it is not PR 1
  material.

## Log

### W1 chime fix (zoompilot) - done
`develop-big-model-fixes` at `cb2d77e03a`: `selfdrived: chime Big Model Ready on the first big frame`.
Diff 8/1 in `selfdrived.py`, new `tests/test_big_model_ready.py` (traces (a) failed load, (b) good
load, big while not alive, re-arm after a fall). Verified by the main agent: 16 selfdrived tests
pass, ruff clean, `big_model_ready_t` and the settling window untouched.

### W2 PR 1 branches - done
- `sp/big-model-fixes` (sunnypilot master base): `659f8f42f2` seen guards (d0115f6a68 + 44d75a6682
  squashed, byte-identical to develop's lines 455-465, new `test_localizer_alerts.py` proven
  non-vacuous against master's file), `0a7979ca19` fetcher fix reworded to the stray-manifest
  fact (docstring too), `5e196cb129` chime fix. 72 passed. Session trailers a subagent added
  were stripped; CLAUDE.md forbids attribution in commits.
- `develop-big-model-fixes`: `cb2d77e03a` chime, `17e3071778` fetcher. 70 passed.

### W3+W4 module API and modeld (jetson-trt) - done
`aaff045066` accelerators: replace the registry with a jetlink module API; `f1b6700403` modeld:
jetlink path beside the native chestnut block. Spot-checked by the main agent: modeld diff vs
develop is the import plus five hunks (21/1), ChestnutState and the fallback untouched; capnp
adds `acceleratorState @4`/`acceleratorName @5` plus the enum, `bigModelLinkLost` pending in W5;
`base.py`/`chestnut.py` gone, `state.py` -> `status.py` (logs telemetry, publishes nothing).
Deviation accepted: the small-model guard is `if not JETLINK:` rather than a sixth
`small_model = None` hunk. modeld_v2 tests keep HEAD's network-free rewrite (not an import
change). 279 passed, validator "cereal compat OK".
`97dc32c9a5` adds `models.helpers.effective_small_bundle()` up front so W6 and W7 can share it.
W5, W6, W7, W8, W10 dispatched in parallel on jetson-trt, each committing only its own paths.

### W5 selfdrived adapter - done
`455c7cea50`: develop's selfdrived plus the chime fix plus import, constructor line and one
`AcceleratorEvents.update()` call (11/1). `bigModelLinkLost @29` added. Design note: the
link-lost edge is tracked only while `acceleratorState != none`, so a chestnut fall never
raises both `bigModelFailed` and `bigModelLinkLost`. 36 passed, validator OK.

### W6 model manager - done
`ef5aa8b5ad`. Every `catalog()` reverted to develop's chestnut_present meaning; kept the
`uses_stock_runner()` short-circuit in `get_active_bundle`, the `maybe_apply_default_model`
skip, `effective_small_bundle` at both qcom reads in `model_info.py`; the `'accelerator'` branch
of `carrying_model()` is guarded on no board fitted. 76 passed. New `TestEffectiveSmallBundle`,
`test_model_info.py::TestCarryingModel`.
Housekeeping debt: the shared index swept W8's staged gitlink/`release.json` deletion and W5's
test files into this commit (git commits the whole index). Content is consistent; split with
`rebase -i` once the branch is quiescent if per-package attribution matters.
Follow-up for the accelerators tree: `accelerators/jetlink/helpers.py:276-278` `active_bundle()`
still falls through to the chestnut slot, which now only ever holds comma-GPU pkls.

### W8 boot, packaging, hardwared, sysctls - done
`4aa02de1a1` jetlinkd owns the VM tuning (16M/8M/128M; record in `/dev/shm/jetlink-sysctl-prev`,
never clobbered, restored in `finally` and on disable mid-run; 6 new tests). `634046e20d`
hardwared +12 (alert, bounded shutdown before DoShutdown), usb.py identical to develop, setup.sh
gated on `JetlinkEnabled` through `openpilot.common.params` (a params library not built yet reads
as off; on AGNOS the updater builds in the staging overlay so the library is normally present at
launch), submodule `jetlink` at `jetlink_repo` = `2d6d85b628`, SConscript candidates collapse to
`#jetlink_repo`. Note for the sunnypilot PR: `.gitmodules` opendbc URL must be sunnypilot's.
`accelerators/jetlink/helpers.active_bundle()` chestnut-slot fall-through kept on purpose: its
docstring explains a catalog bundle shipping an ONNX would be usable, none does today.

### Housekeeping on jetson-trt
- `99bc07bf86`: `modeld/helpers.py` back to develop (HEAD carried a four-line comment there; the
  upstream-owned footprint stays at the five modeld sites).
- `c50443be5b`: `opendbc_repo` pointer moved from the pre-squash `75f9db75b3` to develop's `676fa14b20`;
  `git diff --stat` between the two is empty, so the tree is unchanged and the branch now agrees
  with develop on the submodule.

### W7 UI - done
`bf3c0162b0`: `selfdrive/ui/ui_state.py` is develop plus two hunks (7/1): the accelerator-view
short-circuit at the top of `_update_chestnut_state` and the `usb_unknown` guard. The view, the
5 Hz build and `_accelerator_state()` live in `UIStateSP`; `chestnut_catalog` is gone;
`chestnut_compiled |= model_runner_tinygrad` is gated on `not uses_stock_runner()`. Shared
`accelerator_link.py` feeds both the mici and tici models layouts (toggle + picker + status
note). 146 passed. Follow-up dispatched: collapse the Auto/On/Off link control to on/off, since
"auto" now means off.

### Interim verification on jetson-trt (main agent, after W3-W8)
- Full affected suite (accelerators, models, modeld_v2, selfdrived, sunnypilot selfdrived, ui):
  549 passed, 3 skipped.
- Footprint vs develop, upstream-owned files: .gitignore 1, .gitmodules 3, launch_chffrplus.sh 3,
  custom.capnp 20, params_keys.h 12, modeld.py 21/1, alerts_offroad.json 4, selfdrived.py 11/1,
  ui_state.py 7/1, hardwared.py 12, process_config.py 4. All within the plan's 4.11 estimates.
- `accelerators.` appears at three lines in modeld.py (ready+prepare share one line, the fallback
  `raise` and the SP fields reference `JETLINK` only). The name `jetlink` reaches upstream dirs
  only as the `JETLINK` local, one log string, one hardwared comment and params_keys.h.
- ruff clean on every changed .py file; the whole-tree count (391) is the pre-existing baseline
  with uvx's ruff, develop shows the same class of noise.

### W10 test harness - done
`30d958b560`: `test_native_equivalence.py` lifts modeld's decide/load statements by AST and execs
them under fakes (chestnut fitted: nothing asked of accelerators; no board and no opt-in: only
`ready()` is asked; ChestnutState, the poller wait, the `if CHESTNUT:` body and the fallback are
byte-identical to develop). `test_selfdrived_traces.py` asserts the whole ordered event list for
traces (a), (b) and a jetlink trace (c). `scripts/merge_replay.sh` does a real `--3way` apply.
Merge replay on jetson-trt vs develop: identical failing-file sets for both #38760 and #38684;
the only cost jetlink adds is two extra conflict regions in modeld.py under the #38760 revert
(the `JETLINK =` line and the `elif JETLINK:` load). Neither patch touches `sunnypilot/`, so the
plan's "confined to sunnypilot/" condition is unreachable on any branch that is not clean; the
honest statement is "same manual rebase as develop plus two regions that are ours".
Pinned by a test and worth a decision: a fitted board whose PCIe never trains falls through to
`accelerators.ready()` once (False without opt-in), as the plan's selection line is written.

### Phase C prep
`sp/jetlink` reset onto `sp/big-model-fixes` so PR 2 stacks on PR 1 (the adapter relies on the
chime fix; the fetcher and seen-guard hunks are then already present).

### Bench state before deployment (2026-09-07 05:01 comma time, main agent over ssh)
comma at `2409aa22b` (pre-upstreaming), jetlinkd running, `JetlinkEnabled` absent, `JetlinkModel`
absent, `JetlinkEngineReady` set, UDC `not attached`, and the three sysctls already applied
(16M/8M/128M) with nothing having asked for the link: the plan's "installation alone enables
the feature" finding, live. Jetson `192.168.1.87` does not answer ssh and its ARP entry is
incomplete: off or asleep with no gadget to wake it. Items 9 and 10 of section 6 need it.

### W9 docs - done
jetlink repo `c14c820` (CLAUDE.md, README, `docs/openpilot-integration.md`, `ffs.py` comments),
`04b3ba4` + `8ec176f` (`docs/releasing.md`: the pin is the submodule sha, no lock file). Fork
`b50eb2f6f5` (joining/model_state comments) and `225d0d7328` (backend/warp_cache comments: no
loader thread on the jetlink path; make_model_state runs on modeld's main thread before the
frame loop). CLAUDE.md now carries the measured upstream-owned footprint: 12 files, +99/-3.

### Phase C: `sp/jetlink` built - done
Seven commits on top of `sp/big-model-fixes` (accelerators, modeld, selfdrived, models, ui,
hardwared/boot, tests). Upstream-owned footprint vs PR 1: 11 files, +90/-2, every file within
plan 4.11. Conflicts resolved: custom.capnp ordinals are `bigModelAvailable @26`,
`bigModelLinkLost @27` on master (zoompilot's develop holds @26/@27 already, so the fork uses
@28/@29; the two trees disagree on those two ordinals and always will until one merges the
other); `models/manager.py` has no diff on master (no `default_bootstrap` there, so W6's skip has
no target; it must come back if sunnypilot adds one); fetcher.py is PR 1's only. New-file
headers reworded to "part of sunnypilot" across the seven commits. 501 passed, ruff clean,
imports ok, validator "cereal compat OK". Merge replay: #38760 adds the same two modeld.py
regions as on the fork; #38684 identical to master's own conflicts.

### Section 6 item 8, opt-in inert - PASSED on the comma (side copy at /data/openpilot-upstreaming)
With `JetlinkEnabled` absent: `setup.sh` reads `enabled=0` and exits before `setup_gadget.sh`;
`present()/ready()/uses_stock_runner()` all False, `unavailable_reason()` None,
`get_active_bundle()` None (stock). With `JetlinkEnabled=1` and the venv python on PATH as
`launch_env.sh` provides: `enabled=1`, `setup_gadget.sh` reached. Caveat recorded: the gate
runs `python3`, which must be the venv's (system python3 lacks zmq); launch_chffrplus sources
launch_env.sh first, so it is. Param restored to absent afterwards.
Finding from the same run: `ready()` was True with `JetlinkModel` absent (selected_model()
defaults to the registry) while `uses_stock_runner()` was False (it required JetlinkModel).
Enabled + default model + a custom small bundle would run modeld_tinygrad with a ready Jetson.
Fix dispatched: `uses_stock_runner()` = `enabled()` alone, on both branches and in the docs.

### Section 6 item 11, shutdown bound - PASSED (partial) on the comma
`accelerators.shutdown(timeout=25)` with `JetlinkEnabled` absent: 0.7 ms (one param read). With
`JetlinkEnabled=1` and no Jetson (UDC `not attached`, no dormant marker): 0.00 s, because
`backend.shutdown` skips when `gadget_present()` is False; there is nothing to wake. The 25 s
bound itself (a Jetson known to be there but not answering) needs the bench Jetson and was not
exercised. The core's thread-and-join bound is covered by the unit test in `test_accelerators.py`.

### Not run: section 6 items 9 and 10
Both need the bench Jetson, which is off or asleep with no gadget to wake it (ARP incomplete).
Item 9's unit half (`TestCarryingModel`, `TestEffectiveSmallBundle`) passes; the on-bench half
and the 180 s live bench are outstanding, as is the sysctl-restore line in a real jetlinkd exit.

### Closing state (2026-09-07)
- `uses_stock_runner()` = `enabled()` alone landed on jetson-trt (`16f01e0a5d`), folded into
  PR 2's first and fourth commits, and in the jetlink docs (`9584a60`).
- Submodule pin bumped to jetlink `9584a60` on both branches (docs and comments only).
- Final runs: jetson-trt 556 passed, 3 skipped, ruff clean, 16 commits since the safety tag,
  tree clean. sp/jetlink 501 passed, 3 skipped, ruff clean, 7 commits on PR 1, footprint 11
  upstream-owned files +90/-2. sp/big-model-fixes 72 passed. develop-big-model-fixes 70 passed.
- Nothing pushed. The comma keeps its production install untouched; the side copy at
  `/data/openpilot-upstreaming` (with production's `libparams_c.so` copied in and the launch
  symlinks made) is there for the remaining bench items and can be deleted.
- Left for the user: push and open the two PRs (`docs/pr-sunnypilot-{1,2}.md`), merge
  `develop-big-model-fixes` into develop and `jetson-trt` when ready, run section 6 items 9
  and 10 with the Jetson up, and decide whether the fitted-board-untrained-PCIe fall-through to
  `accelerators.ready()` (pinned by a test) is the wanted reading of the selection line.

### Bench validation 2026-09-07 (after the user's "fully validate" request)
Record: `docs/validation-2026-09-07.md`. Comma updated through git fetch/checkout of the pushed
`jetson-trt` (three reboots: 5d2431150a, 175ceae4e6, 03574347a0), jetlink checkout rsynced to
`/data/openpilot/jetlink_repo`, `JetlinkEnabled=1` set explicitly (the new opt-in rule). 40 of
the 43 rows PASS or PASS-by-test; E2 and the missing-package alert are NOT RUN; B4/C5/C7/C9 are
PARTIAL (unit-pinned, not sampled inside the ui/selfdrived processes). Review findings from
Codex confirmed and fixed: link-loss soft disable was one tick and SP-only; sysctls were
restored at the ignition handoff. Found on the bench and fixed: zero-valued sysctls cannot be
restored by writing 0 (ratio keys), tinygrad's compile pool inherited FIFO 54 on core 7, the
write watchdog likewise, the bench summary crashed on the new state column. Master's own
`bigModelFailed` is also edge-triggered for one tick (same cancel path); not changed here,
flagged for the user.

### Comment style pass (2026-09-07, user request)
Every comment and docstring across all branches rewritten toward sunnypilot's style (one to
three lines, why not what, numbers only when they are the reason, no discovery narrative).
Fork (jetson-trt `1bfd3b600e`, `40ccd65acb`, `fa9814ebc0`, pushed): added comment lines 559 -> 346,
plus module docstrings cut to a paragraph; ported into `sp/jetlink` as fixups of the owning
commits (still seven on the rewritten PR 1); PR 1 wording identical on `sp/big-model-fixes` and
`develop-big-model-fixes`. jetlink repo (`24803a1`, `f00bcda`): 723 -> 473 comment lines, comment
blocks 28 -> 18 words on average. Every changed .py on every branch proven code-identical by AST
with docstrings stripped; non-UI suites green (UI suites cannot open a window in this session:
raylib segfaults in `_calculate_auto_scale`; they passed before the comment-only change).
Comma checkout synced to `819465081d` without a reboot (comments only). Pin f00bcda everywhere.
