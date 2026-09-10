# Status and known limitations

Jetlink is experimental and not a supported public driving release. Bench tests
and individual drives do not establish reliable behavior in every vehicle,
temperature, or failure condition.

## Platform testing

- **Jetson Orin Nano Super, 8 GB:** tested on the car and in recorded-drive
  replay with TensorRT 10.3.
- **Apple silicon:** bench-tested on a 16 GB M1 Pro. CoreML on the GPU meets
  the 50 ms frame budget and loads a prepared model in about 2 seconds.
  tinygrad loads in a second but misses the budget on that machine. See
  [measurements](backends.md#mac-measured).
- **Linux NVIDIA (CUDA laptop):** hardware-tested.
- **Windows WSL2:** implemented, not yet tested on hardware.
- **CPU:** functional testing only.

## Measured performance

Orin Nano Super 8 GB, TensorRT 10.3 FP16, over USB 3, recorded-segment replay:

| Model | GPU inference | Full modeld mean / max | First engine build |
| --- | ---: | ---: | ---: |
| BMRLNAP, 766 MB | 19.8 ms | 31.0 / 32.7 ms | 166 s |
| TGC v2, 766 MB | ~20 ms | 31.1 / 33.5 ms | 166 s |
| Lebowski, 1757 MB | 36.2 ms | 46.3 / 49.5 ms | 290 s |

Full modeld timings include image processing, transport, inference, and output
parsing. The frame budget is 50 ms, so Lebowski leaves little margin. These are
short bench runs and say nothing about sustained performance under heat.

## What still needs validation

Long runs under heat and load, repeated cold starts, cable faults, and recovery.
Link failure falls back to the small model and soft-disables if engaged; the
fallback timing and its effect on control still need qualification.

USB power behavior, voltage dips, suspend and wake, low-battery shutdown, and
long parking periods are not fully qualified. Use separate supplies and read the
[power guide](transport.md#power-requirements) before enabling suspend.

TCP has no client authentication. Use a trusted network. Wi-Fi missed the frame
budget in testing; use USB 3 or wired Ethernet.
