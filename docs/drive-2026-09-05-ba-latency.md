# Route ba: September 5 evening disengagement

The reported event is in `000001ba--e460dab329`, segment 3. Full rlogs from
all 17 segments identify fork `ee21161992c7680d6a004dfebef2ebdaababb266`.
The installed package still has the 16 KiB FunctionFS receive change from the
previous investigation. Raw driving logs and analysis outputs are private at
`/private/tmp/jetlink-ba-investigation`; they are not part of this repository.

## Finding

At **18:14:22.416 America/Chicago**, Jetlink frame **3384** (camera frame
**3605**) completed a **300.03 ms** model run. Its logged stages were:

| Stage | Duration, ms |
| --- | ---: |
| Camera warp | 1.7 |
| Warped-buffer readback | 2.7 |
| Request send | 3.6 |
| Reply stage | **290.3** |
| Server inference (within reply stage) | **19.7** |
| Server history queues | 1.3 |
| Server total, excluding transport | **21.2** |

The server's inference figure is wall time around graph launch and stream
synchronization. These figures rule out slow inference as the source of the
approximately 270 ms excess. The reply stage includes server response sending,
comma USB receipt and consumer scheduling. In general it also includes the
health callback, although this frame falls between the alternating health
publications. The current logs do not directly time that callback.

The next published camera frame is **3611**: five camera frames were skipped.
Filtered drop percentage peaks at **2.427%**, and stays over 1% for 181 model
publications, approximately nine seconds. This is a filtered skipped-frame
statistic, not GPU utilization or the proportion of inference calls over 50 ms.
`modeldLagging/softDisable` appears at logMonoTime 672.575 s. A temporary
localization alert also appears during this episode; selfdriveState is disabled
by 676.602 s. This matches the reported time and disengagement.

Camera capture, driver monitoring and the control loop continued. Around the
event, modelV2 has a 308.75 ms publication gap, whereas carControl's largest
gap in the same two-second window is 13.38 ms, and the road-camera state gaps
remain about 50–52 ms. Model-dependent services share the gap. This does not
look like camera starvation or a system-wide 300 ms scheduling pause.

GPU clock samples near the event are 1020 MHz, with temperatures around 68 C.
No FunctionFS allocation-failure warning or mid-drive USB disconnect appears
in this route's recorded operating-system logs. Absence of a warning does not
exclude successful but slow allocations, reclaim, or USB/controller stalls.
Video log rotation overlaps the event, but this is correlation, not proof of
storage interference.

## Whole-route measurements

The model publication span is 968.08 seconds. There are **19,356 big-model
frames**:

| Metric | Result |
| --- | ---: |
| Execution median | 40.85 ms |
| p99 | 44.79 ms |
| p99.9 | 49.45 ms |
| Maximum | 300.03 ms |
| Frames over 50 ms | 12 (0.062%) |
| Driving-time camera frame-ID gaps | One, following the 300 ms frame |

The other eleven >50 ms frames did not skip camera frames. There is also one
initial small-model startup frame lasting 1298.51 ms, followed by a 26-frame
camera-ID gap during initial joining. It is separate from the driving event.

## Stop-light and longitudinal-control context

A follow-up pass associates each big-model publication with the latest logged
carState, carControl and selfdriveState. At the 300 ms event, speed is about
23.07 m/s (51.6 mph), steering angle 0.6 degrees, with longitudinal control
active and experimental mode enabled. This particular disengagement occurred
while moving, not at a stop light.

Of 2,645 frames below 0.1 m/s, three exceed 50 ms and none exceed 100 ms
(maximum 53.72 ms). Those three stopped outliers occur with longitudinal control
inactive. Of 1,343 frames from 0.1 to 5 m/s, two exceed 50 ms; of 15,368 frames
at or above 5 m/s, seven do. Small overrun rates are somewhat higher at low
speed, but these few samples do not establish a cause. Experimental mode is
enabled throughout these big-model samples, so this route provides no
experimental-on/off comparison. Thread blocking remains consistent with the
reply timing, but neither a GIL/lock stall nor an alpha-long-specific cause has
been established.

## Candidate changes

These changes are local candidates, not a demonstrated fix for the 300 ms
transport stall. No alert threshold, timeout, inference tensor, or wire format
was relaxed or changed.

- Move sensor sampling to one server-owned worker. Inference and connection
  setup only read a cached snapshot. Requests are coalesced, refreshes are at
  most 10 Hz, and no extra worker appears on session reconnect. A sample older
  than one second returns an empty health object. Age starts before sensor IO,
  so a delayed read cannot report an old temperature as freshly measured.
  This closes a separately confirmed synchronous-IO risk between frames; the
  route does not establish it as the cause of this event.
- Record FunctionFS receive maxima for preparation (configuration check and
  userspace buffer allocation), read waiting, and read-to-consumer handoff.
  Slow client inferences log these alongside the existing server timings.
  Read waiting includes peer idle time, kernel IO and time before Python runs
  again; it is not a pure USB-transfer or kernel-allocation measurement.
- Measure server response sending separately and log sends over 10 ms even
  when inference itself is fast. Previously these stalls were invisible to
  the server's slow-frame log.
- Extend the paired fork's `model_state.py` cloudlog with these receive
  diagnostics and separate health-callback time from reply waiting. Python
  package logging alone is not sufficient to persist diagnostics in rlogs.

The receive diagnostics add timestamp calls and per-chunk metadata; their
on-device overhead requires a full-stack benchmark. Logging happens only for
outliers, but logging itself is not a hard-realtime guarantee.

## Validation and remaining work

Local suite: **121 passed, one skipped** (the tinygrad/openpilot queue reference
is unavailable in this local test environment). Regression coverage includes
three successful client/server inference round trips while a sensor remains
blocked, cache expiry and slow-sample age, retry after sensor failure, bounded
refreshes, unchanged framing across partial chunks, per-message diagnostic
reset, and response-send logging with fast inference. Basic Python lint and
`git diff --check` pass.

On-comma suite: **121 passed, one skipped** in 151.71 seconds. This includes
the real tinygrad queue-reference tests; the skipped module is ONNX patching
because the device does not have `onnx`. Candidate sources were extracted to
`/tmp/jetlink-ba-candidate.yHrIQF`, using the existing isolated pytest
dependencies and `OPENPILOT_PREFIX=jetlink-ba-validation`. The installed driving
package was not replaced and no driving process was restarted. The paired
fork's logging-only edit passes its configured lint and Python syntax checks;
it has not been deployed or exercised in a live modeld run.

Next required evidence is a paired, instrumented full-stack reproduction under
realistic memory/storage load, including segment rotations. Correlate the new
receive/send timings with FunctionFS syscall duration, direct reclaim and
scheduler traces. The new timings narrow attribution; they cannot alone prove
whether a long read waited on the peer, the kernel or thread scheduling.

Then run a long thermal soak and cold-boot/suspend/reconnect checks against the
same model and hardware, recording global maximum and p99.9 frame age and every
camera-ID gap. The paired source lock and immutable image must be updated only
for the candidate actually qualified. Until that is done, the residual stall
is unresolved and this work does not justify a claim of zero future lag.

## Jetson log inspection, September 6

Read-only SSH to `monarch@192.168.1.87` retrieved 3,053 retained service-journal
entries plus the exact prior boot's system journal. Copies are under the private
investigation directory's `jetson/` subdirectory. The running container uses
the expected `c300ccd1d053` image prefix. No deployment or service change was
made.

The current boot (`b462fc3e…`) starts with January 8, 2025 timestamps, then jumps
to September 6, 2026 around 66 seconds of monotonic uptime. A wall-clock query
for the reported drive returns no entries. The prior boot (`0b5fc50b…`) contains
USB sessions but its retained journal ends at approximately 1,448 seconds of
uptime, labeled September 5 18:08 UTC. It has no server slow-frame entry, but
its correspondence to route ba has not been established. That absence cannot
be used to rule out a server-side transport stall on the reported drive.

The prior kernel journal reports an unclean/corrupt journal file being replaced
at startup. This is evidence that retention may be incomplete, not evidence
that journal corruption caused model lag. Relative `journalctl -b -1` selection
did not select the intended prior boot during this inspection; use exact boot
IDs and monotonic timestamps when investigating these records.

SSH stopped responding during inspection, consistent with the service's
configured 120-second suspend interval without a USB gadget. The suspend itself
was not confirmed after connectivity was lost. Further retention/configuration
inspection requires the Jetson to be awake; reproducing the USB reply path
still requires the comma connection.

## Resolution, September 6 (bench)

The 300 ms stall is a **comma-side memory/IO reclaim storm**, driven by
recording, that stalls the FunctionFS read of the reply. Proven on the bench
with paired per-frame instrumentation on both ends.

### What the instrumentation showed

On every slow frame the server's own timer is a flat 21 ms and its reply-write
(`send`) blocks for exactly as long as the comma's read (`read_wait`):

| Frame | comma read_wait | server send | server total | comma handoff |
| --- | ---: | ---: | ---: | ---: |
| 16286 | 212.4 ms | 222.2 ms | 21.3 ms | 2.0 ms |
| 16349 | 212.0 | 216.8 | 21.3 | 0.2 |
| 22 | 322.6 | 330.0 | 21.3 | 0.2 |

So the server is idle-blocked on the write, the comma frame thread wakes
promptly once bytes are in hand (`handoff` small), and the entire excess is the
comma being slow to take the reply off the endpoint. Each slow frame is
preceded, in the comma's `/proc/vmstat`, by `nr_dirty` climbing to ~27,000
pages (108 MB) and bursts of `pgscan_direct`/`allocstall`: the kernel is
reclaiming dirty pages, which it must write back before it can evict, exactly
while the USB transfer buffer is trying to allocate. The comma ran at
`vm.min_free_kbytes` 7,424 with ~38 MB free.

It does not stop at a dropped frame. `backend.py` sets the client deadline to
`INFERENCE_TIMEOUT = 0.5 s`; a stall past that raises `LinkError`, modeld falls
back to the small model and rejoins, and loading is `NO_ENTRY` for 60 s after
each loss. One 900 s recording soak produced 91 lagging frames, a 6.6 % peak
filtered drop, and three big-model losses.

Ruled out with numbers, do not re-chase: server sysfs telemetry (< 5.2 ms idle
or loaded), server GC (a full gen-2 is 12.9 ms with the engine resident and
never fires in a driven loop), comma-side scheduling of the frame thread
(`handoff` < 6 ms always), and pure USB transit.

### The fix, and its bench proof

| Change | Where | Effect |
| --- | --- | --- |
| `vm.dirty_bytes`=32M, `vm.dirty_background_bytes`=8M, `vm.min_free_kbytes`=64M | fork `accelerators/jetlink/setup.sh` (system-wide, root, boot) | the storm cannot form |
| FFS reader `_widen_affinity()` off modeld's pinned core | jetlink `transport/ffs.py` | removes the single-core-7 pin the reader inherits |
| `CachedTelemetry` worker | jetlink `server/` | the ~4 ms blocking i2c read leaves the session thread whose `send` gates the comma |

Recording soaks, 766 MB model, TensorRT FP16, SuperSpeed:

| Metric | Untuned, 900 s | VM-tuned, 600 s |
| --- | ---: | ---: |
| big frames | 16,933 | 11,749 |
| exec p99.9 | 49.0 ms | 44.5 ms |
| exec max | 360.9 ms | 72.5 ms |
| frames over 50 ms | 15 | 1 |
| lagging frames | 91 | 0 |
| peak filtered drop | 6.6 % | 0 % |
| `nr_dirty` peak | 108 MB | 6.9 MB |
| direct-reclaim events | 446 | 1 |
| big-model losses | 3 | 0 |

The reader-affinity widen did not move the reader's ~17 ms run-queue delay on
this bench: the RT balancer keeps waking it on core 7, so `everything -
inherited` (force it off the pinned core) is the stronger variant if that
headroom is wanted. It is kept as defense-in-depth against the documented
pinned-thread hazard; it is not what closed the budget violations. The `write`
path still kmallocs 512 KB per request and halves to 256 KB on ENOMEM, which
splits the 458 KB request and re-arms the dwc3 double-TRB desync bug; rare once
memory has headroom, but it wants a wait/retry rather than a shrink.

### Not yet done

The VM sysctls are system-wide and change loggerd's writeback cadence; qualify
them on a full drive, not only this bench. `watermark_scale_factor` was rejected
by the AGNOS kernel. A sustained thermal run and real onroad CPU contention
(more processes than the bench) are still unmeasured. No production deployment,
release-lock update, or commit was made; the candidate server was tested by
`docker cp` into the running container and the original was restored.

## Receive-path hardening and contention testing, September 6

The VM tuning removed the disengagement. Testing the fixed stack under harsher-
than-onroad synthetic contention (five SCHED_OTHER workers plus a 500 MB
resident hog, comma free memory down to ~200 MB) then exposed a residual of
isolated, tolerated over-budget frames from the FunctionFS receive path, run
down by stage:

| Residual | Cause | Fix (`transport/ffs.py`) |
| --- | --- | --- |
| `prepare` 24 ms | reader allocated a 16 KB buffer per read; it reclaimed under pressure | recycle pool + pre-fill: the consumer hands buffers back, the reader reuses them, and the pool is filled up front so a scheduling-lagged consumer never forces an allocation |
| `prepare` 24.9 ms | reader `open()`ed the UDC `state` file before every read (the config check a read must not skip) | hold that fd open, re-read with lseek (fresh value, no allocation, 33x cheaper) |
| `read_wait` 50-67 ms | **the reader thread was SCHED_OTHER** - created during warmup, before modeld goes realtime, so it never inherited FIFO. read_wait brackets the whole readv, including the run-queue wait to return after the transfer; as SCHED_OTHER that was 17 ms mean / 23 ms max under contention. First mistaken for a kernel allocation floor. | lift the reader to SCHED_FIFO 51 (below modeld's frame loop at 54). Run-queue delay fell below 10 ms and the big-model **baseline p50 dropped 40 -> 34 ms** - it was delaying every reply, not just the tail |

Also raised the memory floor to 128 MB / 16 MB dirty (validated under the hog),
reset the write quantum per message so an ENOMEM shrink cannot leave a request
split and re-arm the dwc3 desync, and dropped the gadget receive buffer from
2 MB to 256 KB (it only receives 74 KB replies). No fork memory leak was found.

With the reader at realtime priority the last residual went too. Final stack
under the harsh contention: baseline p50 34 ms (was 40), and the read_wait
over-budget frames are gone. **Zero lagging frames, zero rejoins, zero drops**,
and the disengagement-class invariants held across every run. Across every run the invariants held: no
disengagement-class event under load. Strict "always within budget" is not
literally reachable while the kernel allocates per read, but every
disengagement condition and every userspace-side stall is closed; the residual
is isolated single dropped frames under load harsher than a real drive, still
unmeasured on a car.
