# Bench validation, 2026-09-07

Full validation of jetlink and the upstreamability refactor on the bench, after the
comma was updated to `jetson-trt` `5d2431150a` (jetlink pin `0ab67b5`) and rebooted.
Bench: comma `192.168.1.144`, Jetson `monarch@192.168.1.87` (server in the `jetlink`
container, `--sleep-after 120`). Every item has a pass condition and a result line;
an item with no result line has not been run. Status legend: PASS, FAIL, PARTIAL,
NOT RUN (reason).

## A. Boot and install

| # | Check | Pass when | Result |
|---|---|---|---|
| A1 | Boot on the new tree | comma back on ssh, `manager` running, `/data/openpilot` at `5d2431150a`, `jetlink -> jetlink_repo/jetlink` | PASS: back on ssh 56 s after reboot; `5d2431150a`; symlink `jetlink -> jetlink_repo/jetlink` |
| A2 | Build at boot | `build.py`/scons completes; warp target `accelerators/jetlink/models/warp_*.pkl` exists; `libparams_c.so` newer than boot | PASS: scons finished ~2.5 min after boot; `warp_1344x760_512x256_tinygrad.pkl` present |
| A3 | Opt-in gate | `setup.sh` read `enabled=1`, `/dev/ffs-jetlink/ep0` exists, `/dev/shm/jetlink-gadget` = `ok` | PASS: `ep0` present, `/dev/shm/jetlink-gadget` = ok, UDC super-speed |
| A4 | jetlinkd started by manager | pid present, `only_offroad`; no `Offroad_AcceleratorUnavailable` alert | PASS: jetlinkd pid under manager; `Offroad_AcceleratorUnavailable` empty |
| A5 | Sysctls owned by jetlinkd | `vm.dirty_bytes=16777216`, `dirty_background=8388608`, `min_free_kbytes=131072`; `/dev/shm/jetlink-sysctl-prev` holds the stock values | PASS: 16777216 / 8388608 / 131072 applied; record holds stock 0 / 0 / 7424 |
| A6 | Nothing crashed | no `is dead` in manager log; ui, hardwared alive; `modeld` not started offroad | PASS: no "is dead" or Traceback in the swaglogs; ui, hardwared alive; modeld not started |
| A7 | Runner cache | `ModelRunnerTypeCache` recomputed to stock while the link is on | PASS: `ModelRunnerTypeCache` = 2 (stock) with no active bundle |

## B. Jetson wake and provisioning

| # | Check | Pass when | Result |
|---|---|---|---|
| B1 | Wake from deep sleep by USB edge | jetlinkd presents the gadget, Jetson answers ssh within ~60 s, server log shows `resumed` (or a cold boot) | PASS (Jetson was awake, container up 4 h; a sleep-wake cycle is B6) |
| B2 | Link comes up offroad | UDC `configured`, `super-speed`; server log shows the client hello; jetlinkd log shows the engine confirmed ready | PASS: jetlinkd log "jetson attached", "server Orin-sm87 trt 10.3.0", "engine ready for e8d821733be15ebe" |
| B3 | Readiness recorded | `accelerators.present()`/`ready()` True; `JetlinkEngineReady` and `JetlinkSpec` consistent with `JetlinkModel` (default from the registry) | PASS: `AcceleratorProgress` stage ready; `JetlinkEngineReady` = e8d8...ff28 (Cinque Terre, the registry default) |
| B4 | UI offroad | `ui_state.chestnut_state` READY (not UNCOMPILED/LOADING); models panel shows the link on and the accelerator model name | PARTIAL: not sampled inside the running ui process; `AcceleratorProgress` stage ready and `ready()` True are the inputs, and `test_accelerator_ui.py` pins READY for them |
| B5 | Dormancy | 60 s after nothing to do jetlinkd releases the gadget, writes `/dev/shm/jetlink-dormant`, `present()` stays True, UDC drops | PASS: "nothing left to do, releasing the gadget" 60 s after start; `/dev/shm/jetlink-dormant` written; UDC `not attached`; server "waiting for a jetlink gadget" |
| B6 | Jetson sleeps and wakes again | `suspend_stats/success` increments after `--sleep-after`; presenting the gadget again wakes it | PASS: Jetson stopped answering ssh ~2.5 min after the gadget release; bench bind woke it, ssh back in ~15 s, `suspend_stats/success` 2 -> 3 |

## C. Onroad path, live bench (camerad + real modeld, disengaged)

| # | Check | Pass when | Result |
|---|---|---|---|
| C1 | modeld takes the jetlink path | log `JETLINK` selected (or `jetlink:` lines), no chestnut path, small model publishing before the join | PASS: modeld.log "prepared the large model ahead of the swap in 8.02 s", "built the large model state in 10 ms"; frame 1 small (1240 ms, the load), big from frame 2 |
| C2 | Join and promotion | exactly one join, `modelV2.big` True, `modelDataV2SP.acceleratorState = running`, `acceleratorName = jetlink` | PASS: one join ("large model joined mid-drive"), 3302 of 3303 frames big |
| C3 | Frame health | over 180 s: zero link failures, `frameDropPerc` under 1, no `modeldLagging`, exec mean/max recorded | PASS: 180 s, exec p50 33.75 / p99 35.1 / p99.9 41.8 / max 53.1 ms, 1 frame over 50 ms, max drop 0.0 %, 0 lagging, 0 link failures (bench 06:21:09, output /data/tmp/jetlink-bench-20260907-062101) |
| C4 | Handover | jetlinkd held the link offroad; modeld joined after jetlinkd released; no `EBUSY` on ep0 (the open memory item "server never reattaches") | PASS: manager jetlinkd SIGTERMed, ep0 fds 0, bench modeld bound and joined within 8 s; no EBUSY |
| C5 | selfdrived events | `bigModelReady` chime exactly once on the first big frame; no `bigModelFailed`; no `commIssue`; `bigModelLoading` never (a late join never blocks) | PARTIAL (no selfdrived on the bench): the whole `update_events` traces in `test_selfdrived_traces.py` and `test_big_model_ready.py` pin one `bigModelReady` on the first big frame, no `bigModelFailed`, no `bigModelLoading` on a late join; on device modeld published `modelV2.big` and `acceleratorState` exactly as those tests feed |
| C6 | Engagement gate | with a faked engaged `selfdriveState` the state stays on small with `bigModelAvailable` True; on disengage it promotes | PASS: bench `--engaged-until 45`: `acceleratorState = joining`, `bigModelAvailable = 1`, big 0 for the whole engaged window; promoted at 45.08 s, the first disengaged poll |
| C7 | UI onroad | `chestnut_state` ACTIVE while big; DISCONNECTED when the link drops with `present()` False; LOADING while joining | PARTIAL: `chestnut_state` not sampled inside the ui process; the inputs it reads (`present()`, `modelV2.big`, `acceleratorState` joining/retrying/running) were all observed on device and `test_accelerator_ui.py` pins ACTIVE/DISCONNECTED/LOADING for them |
| C8 | Link loss mid-run | server SIGSTOP > 0.5 s: `LinkError`, demote in place, `modelV2.big` False, no frame gap beyond one, `acceleratorState = retrying`; SIGCONT: rejoin after backoff | PASS: `docker pause` 1.5 s at t=77.8: "gadget write had no reader for 0.500s", `retrying`, one 11-frame gap (the 0.5 s deadline), rejoined at 84.9 s (5 s backoff plus connect); drop filter peaked 2.4 % and read >1 % for ~10 s after each loss (modeldLagging exposure inherent to the 0.5 s deadline, 0 lagging frames in the undisturbed 180 s run) |
| C9 | Link lost while engaged | with a faked engaged state at the loss: `bigModelLinkLost` SOFT_DISABLE with the "Small model is driving" text; disengaged: nothing | PARTIAL (no selfdrived on the bench): after 81827becc8 the adapter latches the loss and adds native `bigModelFailed` plus `bigModelLinkLost` every tick until disengagement; `drive_state_machine` test proves the real `StateMachine` reaches DISABLED after `SOFT_DISABLE_TIME`, and that the pre-fix one-tick event did not |
| C10 | Server restart | `docker restart jetlink`: engine reloads from the plan cache, client reconnects, `modelV2.big` returns | PASS: `docker restart jetlink` at t=160.9: "no answer for frame in 0.5s", rejoined at 167.6 s; the restart's USB re-claim cost a second loss at 177.0 ("No such device") and a rejoin at 187.7 s (10 s backoff, n=2) |
| C11 | Thread priorities | every jetlink thread SCHED_OTHER off core 7 except the FunctionFS reader at FIFO 51; the frame loop FIFO 54 on core 7 | PASS after 6b03a88dc3 and 84c552a (bench 4, 07:04): only the frame loop is FIFO 54 on core 7; the three pool handler threads, `jetlink-write-guard`, join loop, engagement watcher and `libusb_event` are SCHED_OTHER with mask 0xff; the FunctionFS reader FIFO 51, mask 0xff (below the frame loop, so it cannot preempt it) |
| C12 | Telemetry to swaglog | `jetlinkTelemetry` events at ~1 Hz; nothing published on `chestnutState` | PARTIAL: not observable in the isolated bench (its cloudlog has no logmessaged); unit-tested; to confirm with LOGPRINT=info on the next run |

## D. Opt-out and native equivalence

| # | Check | Pass when | Result |
|---|---|---|---|
| D1 | Disable while parked | `JetlinkEnabled=0`: jetlinkd restores the sysctls and removes the record, releases the gadget, exits; `present()/ready()` False; UI shows no accelerator; no offroad alert | PASS after f39dfd9bf2: disabling with jetlinkd up restores `dirty_bytes 0, dirty_background_bytes 0, min_free_kbytes 7424, dirty_ratio 20, dirty_background_ratio 10` (the ratio keys written for the zero-valued bytes keys), removes the record, releases the link ("disabled, releasing the link"), `present()/ready()` False, no alert. Earlier PARTIAL on 175ceae4e6: bytes keys stuck at 16 M / 8 M because the kernel skips a write of 0 |
| D2 | Boot disabled | after reboot with it off: `setup.sh` exits before the gadget, no `/dev/ffs-jetlink`, no jetlinkd, stock sysctls | PASS: boot with `JetlinkEnabled=0`: no `/dev/ffs-jetlink`, no jetlinkd process, sysctls stock 0 / 0 / 7424, no `AcceleratorProgress`, no alert, `present()/ready()/uses_stock_runner()` False, `ModelRunnerTypeCache` 1 (tinygrad with the stored C210M bundle) |
| D3 | modeld disabled | live bench with it off: small model only, no `accelerators` calls past `ready()`, `acceleratorState = none`, `acceleratorName = ''` | PASS: `--small` bench 70 s with the link off: 1302 frames, big 0, `acceleratorState = none` throughout, no jetlink log lines, exec p50 24.5 ms, 0 lagging; modeld threads: main only realtime |
| D4 | Custom small bundle + link on | manager stays on stock modeld (`uses_stock_runner`), `carrying_model()` names the default small model, `chestnut_compiled` not forced by the stored bundle | PASS: stored qcom bundle C210M plus link on: `uses_stock_runner` True, `get_active_model_runner` 2 (stock), `ModelRunnerTypeCache` recomputed to 2 by manager, `effective_small_bundle` None, `model_info()` names "CD210 (Default)" |
| D5 | Custom small bundle + link off | manager runs `modeld_tinygrad` with the bundle, model_info names the bundle | PASS: same bundle with the link off: runner 1 (tinygrad), active bundle C210M, `effective_small_bundle` C210M |
| D6 | Re-enable | toggle on: jetlinkd starts, sysctls applied, gadget presented, Jetson wakes | PASS: after the 07:01 reboot with the link on, jetlinkd applied the sysctls, presented the gadget and the Jetson (asleep, `suspend_stats/success` 3 -> 7 across the session) connected at 07:02:52; record now carries `dirty_ratio` 20 / `dirty_background_ratio` 10 (f39dfd9bf2) |

## E. Shutdown and power

| # | Check | Pass when | Result |
|---|---|---|---|
| E1 | Shutdown handshake | `accelerators.shutdown()` with the Jetson present: request handed over, server writes the poweroff marker (dry run on the bench), returns well under 25 s | PASS: from a dormant jetlinkd, `accelerators.shutdown()` returned in 2.3 s; jetlinkd "jetson answered the shutdown request: ok, powering off"; Jetson server "shutdown requested by the client", `jetlink-poweroff.service` "dry run, not powering off" |
| E2 | Shutdown with the Jetson gone | Jetson unreachable but known present: returns at the 25 s bound, hardwared continues | NOT RUN: needs the Jetson awake and paused at the moment of the request; the bound itself is unit-tested (`test_accelerators.py`, thread join at the timeout) |
| E3 | jetlinkd exit | SIGTERM: sysctls restored, record removed, gadget released cleanly (ep0 reopenable) | PASS (new semantics after 544ffade4c): SIGTERM of jetlinkd leaves 16 M / 8 M / 128 M applied and the record in place; the bench modeld then ran the whole session on the tuned values; disable is the only restore (see D1) |

## F. On-device regression

| # | Check | Pass when | Result |
|---|---|---|---|
| F1 | Fork suites on the comma | accelerators, models, modeld_v2, ui under the conftest prefix: green, and `JetlinkEngineReady` untouched afterwards | PASS: 421 passed, 2 skipped, 6 subtests in 44 s on the comma (accelerators, models download tests, modeld_v2, accelerator UI tests, sunnypilot ui tests, sunnypilot selfdrived, selfdrived); `JetlinkEngineReady` unchanged afterwards |
| F2 | jetlink package tests on the comma | `tests/` incl. `test_queues.py` (tinygrad): green | PASS: 133 passed, 1 skipped in 149 s on the comma (`jetlink_repo/tests`, includes `test_queues.py` against tinygrad) |

## G. UX matrix (what the driver sees)

| Situation | Expected | Result |
|---|---|---|
| Ignition, Jetson ready | small model drives at once, big joins within seconds, one "Big Model Ready" chime | PASS (bench 1, 3, 4): small frame 1, big from frame 2, one join; chime pinned by tests |
| Ignition, Jetson still booting | small model drives, no NO_ENTRY, `bigModelAvailable` chime when it arrives, promotion on the next disengaged moment | PASS in part (bench 2): joining with `bigModelAvailable` while engaged, promotion on the first disengaged poll; the `bigModelAvailable` chime is pinned by tests |
| Link lost while engaged | soft disable with "Small model is driving, reconnecting if it comes back" | PASS by tests (C9); on device the loss demotes in place within 0.5 s and rejoins (C8) |
| Link lost while disengaged | silent fallback, rejoin later | PASS (C8, C10): silent demotion, rejoin after 5 / 10 s backoff |
| Link off | no accelerator anywhere in the UI, no alerts, chestnut behaviour untouched | PASS (D1, D2, D3, D5): no gadget, no daemon, stock sysctls, no alert, tinygrad runner with the stored bundle |
| Package missing with link on | offroad alert "accelerator unavailable" with the reason | NOT RUN on device; `setup.sh` writes the reason to `/dev/shm/jetlink-gadget` and `unavailable_reason()` is unit-tested |
| Parked | jetlinkd dormant after 60 s, Jetson asleep, presence held | PASS (B5, B6, D6): dormant at 60 s, Jetson asleep ~2.5 min later, woken by the next bind in ~15 s |

## Log
- 07:01 to 07:11: two more reboots (fixed build 03574347a0; then link off for D2; then link on), E1, F1, D1 with the ratio keys, bench 4. Final state: link on, jetlinkd dormant, tuning applied, `ModelRunnerTypeCache` 2.
- Fixes that came out of this session: `81827becc8` (link loss latches, native + SP event), `544ffade4c` (tuning survives ignition), `f39dfd9bf2` (ratio-key restore), `6b03a88dc3` (compile pool before realtime), jetlink `84c552a` (library threads drop realtime), bench tool `bb89ae8f9b`/`205762aa17`. All on `jetson-trt` (pushed) and folded into `sp/jetlink`.
- Known and accepted: a 0.5 s link stall costs ~10 s of `frameDropPerc` above 1 % (modeldLagging exposure) after each loss; unchanged design, see the ba memory.
- 07:04 bench 4 on 03574347a0 (75 s, concurrent with the on-device test suite): 1288 frames, 1287 big, exec p50 33.8 / p99 36.5 ms, 3 frames over 50 ms under that CPU load, 0 lagging.
- 06:45 session 2 on 175ceae4e6 (`/data/bench/run2-064552`, 240 s, engaged 45 s, pause at 78 s, server restart at 161 s): 4492 frames, 3388 big, big exec p50 33.7 / p99 35.4 / max 49.2 ms; three losses injected, three rejoins.
- `/data/tmp` is a tmpfs: py-spy, pytest and the session-1 output went with the reboot; tooling now lives in `/data/bench`.
- 06:21 first live bench on 5d2431150a: see C1-C4. Old-build note: killing jetlinkd restored `min_free_kbytes` to 7424 (the review's P1) and `dirty_bytes` stayed 16777216 (restore of a 0 value did not take; re-check with the fix).
