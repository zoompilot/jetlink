# September 5 drive latency investigation

Analyzed comma revision `3e7e276612` with Jetlink `2721e42`. Downloaded
rlogs/qlogs for routes `000001a8` through `000001b0` and retained swaglogs from
`comma@192.168.1.144`. The Jetson was unavailable. Raw logs and the analysis
scripts are local at `/private/tmp/jetlink-drive-20260905`; do not publish these
private driving logs.

## Measurements

Percentiles below use individual `modelV2.modelExecutionTime` samples, not
rolling-window averages. Counts include join frames. Route duration is measured
between its first and last model messages.

| Route | Duration | Big frames | Median / p99 / maximum, ms | Big frames >50 ms | Maximum filtered drop % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000001a8--0b6d9cfbdd` | 22.6 min | 27,073 | 35.69 / 36.64 / 335.34 | 18 | 2.59 |
| `000001a9--1c538f6242` | 7.0 min | 8,199 | 35.78 / 36.71 / 301.49 | 9 | 2.43 |
| `000001aa--0ccc6d9c58` | 14.6 min | 11,629 | 37.45 / 40.02 / 326.19 | 19 | 3.13 |

The later `000001af` and `000001b0` routes ran only the small model: 24,830
frames combined, no lag alerts, no frames exceeding 50 ms except each process's
initial JIT frame. Their p99 execution times were 31.69 and 32.10 ms.

The large model is generally fast enough. Across the three big-model routes,
46 of 46,901 big frames exceeded 50 ms (0.098%). This is not the same statistic
as the displayed drop percentage: modeld filters skipped camera frame IDs with
a ten-second time constant. A single 300 ms stall can skip five frames and
produce several seconds above selfdrived's 1% lag threshold. All 24 frame-ID
gap events across these three routes followed execution exceeding 50 ms;
that count also includes startup/small-model gaps.

Most logged slow frames are dominated by reply waiting, with a few dominated
by sending. Warp time stays around 1.5-1.8 ms. Reply waiting includes comma-side
receive delays; it is not a measurement of GPU execution.

Stop/turn correlation is mixed: route a8 has no >50 ms frames with absolute
steering angle above 30 degrees, whereas a later aA cluster occurs stopped and
another outlier occurs turning. Do not infer scene-dependent model execution
from the driver's observation alone. Alpha longitudinal has not been isolated
by a controlled comparison.

## Confirmed fault: costly FunctionFS receive allocations

The recorded `operatingSystemLog` messages contain:

```
openpilot.selfd: page allocation failure: order:6, mode:0x24040c0(GFP_KERNEL|__GFP_COMP)
  __alloc_pages_nodemask
  kmalloc_order
  kmalloc_order_trace
  __kmalloc
  ffs_epfile_io
  ffs_epfile_read_iter
```

Route a9 reports this around logMonoTime 1984.511 s. Model frame 4806 publishes
at 1984.767 s with execution time 288.53 ms; the following frame is 4811,
showing five skipped frames. Route aA has further order-6 and order-5 failures.
Journal ingestion can delay timestamps: not every warning coincides with the
nearest model message. A successful but slow allocation need not warn at all.

The AGNOS kernel source in `drivers/usb/gadget/function/f_fs.c` confirms that
the synchronous endpoint path calls `kmalloc(data_len, GFP_KERNEL)` before
posting each read. A userspace bytearray does not preallocate that kernel
buffer. The existing 256 KiB read requests require order-6 contiguous pages
on this 4 KiB-page device. Waiting until ENOMEM to halve the request is too
late to preserve that frame's deadline.

Changed the initial receive request to 16 KiB (order 2), below Linux's costly
allocation threshold. Messages are still reassembled by the stream transport;
the wire protocol and model tensors are unchanged. This reduces fragmentation
exposure, not all possible memory-reclaim latency. A FIFO regression checks an
entire 73,808-byte output across bounded, packet-aligned reads.

## Diagnostic improvement

The server already returns GPU/queue/total execution timings on every inference
response. The client retains them, but the old slow-frame log omitted them.
The fork now includes them and logs at 50 ms rather than 80 ms. The field named
GPU time is measured by the server's wall clock around graph launch and stream
synchronization, not a CUDA-event measurement; it can include host scheduling.
Server total excludes USB receive/send. This lets the next drive separate
server execution from transport/receive delays without retrieving the Jetson.

## What is not established

- Not every outlier has been attributed to allocation stalls.
- Sampled Jetson clocks were 1020 MHz near the slow frames, with temperatures
  approximately 64-72 C. This argues against sustained clock throttling, but
  samples cannot exclude brief power/thermal events.
- Cable integrity is not proven. There is no justification to replace the
  cable solely from these timing logs.
- The write path still requests up to 512 KiB and can suffer similar
  allocation latency. Smaller writes need an end-to-end throughput test and
  review of the persistent write watchdog, signal masking, and burst
  alignment. Do not change them blindly along with the receive fix.
- The receive fix does not change tensors or alert thresholds. The accompanying
  transition changes (disengaged upgrades, reset on fallback, removal of fault
  suppression) need separate qualification; see `release-readiness.md`.

## Release gate

These are candidate changes, not production validation. With the Jetson back:

1. Compare original and bounded-read implementations using the same cached
   model, USB cable, and full live modeld/camerad stack. Include long-running
   onroad-equivalent memory pressure and record global p99.9/max, not only mean.
2. Trace FunctionFS syscall time, allocator reclaim/compaction, scheduling,
   and server timings around every >50 ms frame. Address the remaining write
   allocation path if implicated; measure its throughput tradeoff.
3. Repeat after suspend/resume and cold boot, plus USB disconnect and server
   restart. Verify small-model fallback and truthful UI states.
4. Only after bench validation, perform supervised vehicle testing across
   stop/go, turns, and sustained thermal load. Preserve lag alerts: a device,
   cable, or server can fail, so absence of warnings can never be guaranteed
   by suppressing the safety signal.

The changes are not committed, pushed, or deployed to the comma by this
investigation. Its installed driving code remains unchanged.

Validation completed locally: 83 Jetlink tests passed, one skipped; the ten
FunctionFS tests passed independently. Basic Jetlink lint, the fork's configured
lint for the modified model state, and both repositories' diff whitespace checks
passed. These are software regression checks, not hardware latency validation.
