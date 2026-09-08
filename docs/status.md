# Status and known limitations

Jetlink is experimental and is not a supported public driving release. Successful
bench tests and individual drives do not establish reliable behavior in every
vehicle, temperature, or failure condition.

## Platform testing

- **Jetson Orin Nano Super, 8 GB:** tested on the car and in recorded-drive replay
  with TensorRT 10.3. See the [recorded timings](../README.md#performance).
- **Apple silicon:** bench-tested on a 16 GB M1 Pro. CoreML GPU met the 50 ms
  frame budget in a short test, but model sessions took about 9 minutes to load.
  tinygrad exceeded the budget; the Neural Engine missed some deadlines at 20 Hz.
  See [platform measurements](platforms.md#mac-measured).
- **Linux NVIDIA and Windows WSL2:** implemented paths that have not yet been
  tested on hardware.
- **CPU:** available for functional testing, with no real-time performance claim.

## What still needs validation

Long runs under heat and load, repeated cold starts, cable faults, and recovery
need further testing. Switching to the large model requires disengaged controls.
Link failure falls back to the small model and causes a soft disable if engaged;
fallback timing and its effect on driving control still need qualification.

USB power behavior, voltage dips, suspend/wake, low-battery shutdown, and long
parking periods are not fully qualified. Use separate supplies and read the
[power guide](transport.md#power-requirements) before setting up optional suspend.

TCP has no client authentication. Use a trusted network. Wi-Fi missed the frame
budget in testing; use USB 3 or wired Ethernet.
