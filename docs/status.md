# Status and known limitations

Jetlink is experimental and not a supported public driving release. Hardware
tests and individual drives do not establish reliable behavior in every vehicle,
temperature, or failure condition.

## Platform testing

Support in the installer does not mean a platform has been tested in a vehicle.

- **Jetson Orin Nano Super, 8 GB:** tested in a vehicle and in recorded-drive
  replay on JetPack 6.2 with TensorRT 10.3. JetPack 7.2 (TensorRT 10.16, the
  CUDA 13 image) is not yet tested on hardware: the image builds and its
  TensorRT loads, and the installer is tested against stand-ins.
- **Apple silicon:** tested on a 16 GB M1 Pro. CoreML with the Neural
  Engine and the GPU averages 29 to 32 ms a frame, and the GPU
  alone averages 44 ms, against a 50 ms budget. Some runs had individual
  frames over the deadline.
  tinygrad loads in a second but misses the budget on that machine. See
  [measurements and test conditions](mac-performance.md).
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
parsing. The frame budget is 50 ms, so Lebowski leaves little margin. These
short tests do not establish sustained performance at high temperatures.

## What still needs validation

Further testing is needed for sustained heat and load, repeated cold starts,
cable faults, and recovery. Link failure falls back to the small model and
soft-disables if engaged; the fallback timing and its effect on control still
need validation.

USB power behavior, voltage dips, suspend and wake, low-battery shutdown, and
long parking periods are not fully validated. Use separate supplies and read the
[power guide](transport.md#power-requirements) before enabling suspend.

TCP has no client authentication. Use a trusted network. Wi-Fi missed the frame
budget in testing; use USB 3 or wired Ethernet.
