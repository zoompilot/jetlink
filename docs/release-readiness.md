# Jetlink release-readiness review

September 5, 2026. Reviewed the Jetlink client/server/USB transport, the fork's
accelerator seam, joining state, modeld integration, selection, health reporting,
startup packaging and power lifecycle. This is a code and recorded-drive audit,
not a safety certification or completed hardware qualification. The Jetson was
offline. Unrelated Mazda/opendbc changes were not changed or qualified here.

## Verdict

Suitable for continued development and controlled validation, not a supported
public driving release yet. The architecture has useful boundaries: inference
only on the Jetson; camera capture, calibration, parsing, control and panda stay
on the comma. A cached engine and small-model fallback are good starting points.
However, a successful drive and fast mean inference do not establish safe
transitions, bounded failure handling, or noninterference with the other processes.

## Concrete fixes prepared

Changes are recorded locally with this review; none were deployed. The paired
fork pins the Jetlink source through `release.json`; see `releasing.md`.
Focused tests cover the transport, client/server, accelerator transitions,
fallback reset and model selection. This does not include a complete openpilot
integration/safety suite or any new hardware test. See validation results below.

| Defect | Change | Regression coverage |
| --- | --- | --- |
| Costly contiguous FunctionFS read allocations stalled recorded drives | Start at 16 KiB instead of 256 KiB | Full output reassembly across bounded reads |
| Engagement watcher only reconsidered liveness when messages updated | Recheck seen/alive/valid each poll and expire its snapshot after 250 ms | Missing/invalid streams and a stalled watcher |
| `enabled=False` could be mistaken for inactive control under MADS | Consider `carControl.latActive` and `longActive` too | Independent lateral/longitudinal activity |
| Unsolicited replies could keep `_expect` running after its deadline | Enforce absolute expiry before each receive | Continuous stale replies and progress messages |
| Backend `prepare()` could raise before the small model was loaded | Guard preparation through the accelerator API | Failing and successful preparation |
| Selected model and cached spec could refer to different models | Check identity in readiness and link setup | Mismatched selection refuses cached engine |
| Failed preparation could defer GPU warp setup onto an onroad swap | Fail startup preparation; reject unprepared geometry | Preparation fails before join threads start |
| Successful inference replies were not checked against requested frame ID | Validate frame identity and payload length; latch failed stream | Wrong ID, truncated header/output, refusal to reuse |
| Peer-provided model identity could escape cache directories | Require lowercase SHA-256 before constructing cache paths | Relative/absolute paths and malformed hashes |
| Upload chunks could write beyond declared model size | Validate offset header and extent before opening file | Truncated header and oversized extent |
| Server timing information was lost after every frame | Include existing timings in slow-frame logs; threshold 50 ms | No protocol or numerical-output change |
| A stopped car could upgrade while control remained active | Require fresh, fully disengaged control | Engaged standstill, MADS, expired status |
| Failure teardown blocked the frame thread | Retire links on the background owner before reconnecting | Blocked close does not prevent small-model frame |
| Small-model history froze while the accelerator ran | Pre-capture an in-place zero-history reset for fallback | Repeated CPU-backed resets, buffer identity preserved |
| Per-write Timer creation and cancellation race | One persistent watchdog per link; expiry permanently invalidates it | Reuse, cancellation, blocked abort and deadline race |
| Rejoin could suppress unrelated faults for five seconds | Remove accelerator-specific communication/localization grace | Existing state/alert tests partly blocked by local native Params build |
| Jetlink selection could retain a custom small-model runner | Invalidate runner cache and select stock runner while configured; Chestnut takes precedence | Stored QCOM/Chestnut selections preserved; offroad-only UI mutation |
| Copied client and floating server tag could drift across restarts | Startup source lock, immutable server image ID, protocol version 2 | Source modification/addition/deletion and mixed-version rejection |

The receive change still needs hardware throughput/jitter validation. Smaller
buffers reduce allocation risk but increase syscall count. It is not evidence
that every observed stall has been eliminated.

## Outstanding release blockers

### Model transitions and controls

Upgrades now require fresh, fully disengaged controls. Failure fallback resets
the small model to startup-like zero history. This prevents resuming frozen
observations, but does not establish a smooth or safe closed-loop takeover.
Measure QCOM reset latency and controller-relevant outputs over recurrent
sequences; test braking, turns and driver takeover. Do not call it seamless.

Accelerator-specific fault suppression was removed, including the five-second
post-ready grace period. Lag alert thresholds were not relaxed. Inject unrelated
communication/localization failures during startup and rejoin on the full stack.

### Worst-case frame time and isolation

- Inference has a 500 ms failure budget, not a 50 ms execution guarantee.
- Demotion now defers close to the background owner. Verify actual USB unbind
  behavior and small-model reset cost under kernel/controller faults.
- Endpoint opening has its own ten-second wait, outside the write watchdog.
  This normally occurs on the join thread, but timeout contracts should cover
  all paths rather than depend on the expected call sequence.
- Outgoing USB requests still use large contiguous allocations. Measure before
  selecting a smaller write quantum; retain burst alignment and signal masking.
- The persistent watchdog drops realtime priority and refuses reuse after
  expiry. Measure scheduling behavior and partial-write/abort races on-device;
  a userspace watchdog cannot guarantee a bound on a wedged kernel operation.
- Server telemetry performs sysfs reads on the session thread after a reply,
  delaying when the next read is posted. Consider a bounded-rate cached worker;
  prohibit overlapping refreshes and expire stale health data.
- Startup still waits for accelerator preparation before publishing the first
  small-model frame. A separate loader thread is not asynchronous startup when
  the main thread immediately joins it.

Measure CPU time, scheduler wait, context switches, RSS, allocations, camera
drops, and control/driver-monitoring timing with Jetlink off, absent, joining,
active, failing and recovering. Shared memory bandwidth, GPU use, interrupts and
USB work mean literally zero interference is impossible; require no measurable
regression against a declared budget.

### UI, identity and compatibility

The selector's cached inventory is historical, not live readiness. Display
selected model separately from the model actually producing outputs. Distinguish
absent, asleep, connecting, building, ready-to-switch, running and fallback.
Transitions must be driven by one versioned status snapshot, not combinations
of independently written params and old telemetry.

Runner precedence now preserves Chestnut's catalog and uses stock small-model
execution for configured Jetlink; disabling Jetlink restores the stored QCOM
choice. Validate those combinations with real compiled bundles and reboot.

Protocol version and ONNX hash do not fully express compatibility with a changed
warp implementation, parser, queue semantics or recurrent layout. Negotiate a
model ABI and capabilities, validate shapes/slices before allocation, and test
mixed-version client/server combinations and rollback.

### Deployment, power and security

- The fork now pins client source and the service requires an immutable image
  ID. These are installation guards, not artifact signatures or a complete
  reproducible build. Still pin dependencies, JetPack/TRT, model ABI and build
  artifacts together; archive and qualify the actual server image on Jetson.
- Verify clean installation, interrupted updater/build, rollback, corrupt plans,
  full disk and incompatible caches on fresh hardware, without manual rsync.
- USB sink/power-source startup behavior remains unresolved. Qualify a cable,
  port, supply and backfeed/brownout-safe hardware arrangement.
- Test all boot orderings, ignition bounce, suspend/wake failures, low battery,
  long parking, shutdown acknowledgment and parked current. The prior poweroff
  dry-run and bench sleep tests do not qualify the full car power lifecycle.
- TCP defaults to all interfaces and does not authenticate clients. Do not
  expose it to untrusted networks. Production should default to local/USB
  operation; authenticated remote operation requires a separate threat model.
- Cache-path/upload validation is only part of input hardening. Bound declared
  model sizes, shapes, progress/telemetry and build resource use; fuzz framing,
  truncation, invalid metadata, replay and interrupted uploads. Review the
  root container and writable power-control mounts for least privilege.

## Making upstream acceptance more plausible

1. Propose a narrow interface first, based on actual Chestnut and Jetlink
   requirements. Move shared inference abstractions out of the fork-specific
   model-manager/UI layer. Keep a static explicit backend list rather than a
   general plugin framework.
2. Extract Chestnut with behavior-preserving tests before changing loading or
   failure policy. Require Chestnut present/absent tests to pass without Jetlink
   installed and with a deliberately broken Jetlink backend.
3. Specify typed model choices, state, capabilities and runtime context instead
   of optional duck-typed methods and unstructured dictionaries. Pass current
   controls context from modeld rather than duplicating subscriptions forever.
4. Preserve existing cereal field meanings, not merely field names/numbers.
   Using `pcieLtssm=0x78` to mean a healthy USB Jetson is an adapter today, not an
   agreed upstream schema. Discuss additive accelerator identity/status fields;
   keep a compatibility adapter for existing UI consumers.
5. Submit independent bug fixes, failing tests and measured performance changes
   separately. Keep the Jetson server, installation, model catalog and fork UI
   separate from the minimal inference integration. Exclude Mazda tuning and
   unrelated submodule changes from the accelerator submission.

Upstream's contribution guide favors small, well-tested fixes and explicitly
lists large PRs and most new features as poor candidates. It also requires
benchmarks for optimizations and preserving stock logging semantics. Acceptance
of the whole feature must be discussed, not assumed:
https://docs.comma.ai/CONTRIBUTING/

## Validation required before public driving support

Local validation limitations: the complete accelerator suite crashed in native
code, also reproduced running `test_jetlinkd.py` alone. Selfdrived tests had
10 passes and two failures because the local Params library rejected existing
`Offroad_Chestnut*` keys. Neither is counted as a passing integration run.
Rebuild native dependencies in a matching fork environment and rerun before
deployment. Focused passing suites are not a substitute for this gate.
The initial 30 selector/UI tests passed; subsequent runs (including one new
offroad-toggle test) crashed in raylib font initialization before tests ran.
The new UI test therefore remains unverified in this environment.

Completed local runs: 104 Jetlink tests passed, one skipped; 157 focused
accelerator/model-manager tests passed, one skipped. Jetlink basic lint,
configured fork lint for all changed Python files, shell syntax and whitespace
checks passed. The small-model reset test uses CPU tensors, not QCOM hardware.

1. **Software:** protocol fuzz/property tests, bounded resource tests, state-machine
   invariants, both backend contract suites, model-manager/UI/process startup
   regressions, and deterministic tests for every timeout/close race.
2. **Numerical:** independent reference inference and full modeld replay per
   supported model/ABI. Include recurrent sequences, resets, dropped frames,
   reconnects and actual controller-relevant outputs. High correlation alone
   can hide biased or discontinuous acceleration/curvature.
3. **Hardware:** repeated cold boots and fault injection; multi-hour hot/cold
   full-stack soaks under realistic memory/storage/network load. Record global
   tail latency and frame age, not just GPU time or rolling means. Every missed
   deadline needs an explanation and a validated safe response.
4. **Controls:** software-/hardware-in-the-loop transition scenarios, including
   braking, stopping, restart, curves, MADS, stale messages and unrelated process
   faults. Preserve driver override, driver monitoring and panda safety limits.
5. **Vehicle/staging:** supervised, controlled validation followed by a limited
   documented test cohort, reproducible artifacts, rollback and issue reporting.
   Publish supported hardware/models and known limitations before wider use.

These gates are proposed for Jetlink, not a claim to reproduce comma's entire
internal release process. comma documents software-in-the-loop,
hardware-in-the-loop and vehicle testing before releases:
https://docs.comma.ai/concepts/safety/
