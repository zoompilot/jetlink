# Performance and operating limits

Jetlink is experimental. If the link drops while engaged, the comma
soft-disables and tells you to take over. See [daily use](using-jetlink.md)
for model switching and reconnection behavior.

<a id="status-and-known-limitations"></a>
<a id="platform-testing"></a>

For hardware and software requirements, see [Jetson setup](jetson.md),
[the Mac guide](macos-app.md), or [PC setup](platforms.md).

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

For Mac latency, preparation times, and backend comparisons, see
[Mac performance](mac-performance.md).

iPhone 17 Pro, experimental ([Jetlink for iPhone](../ios/README.md)): the
766 MB model on the Neural Engine at 17.7 ms mean, 21.5 ms p99 a frame, parity
gate passed. Over one USB cable, zoompilot's parked live bench ran every frame
on the phone at 39.1 ms p99. Driven once, about 30 minutes; the phone slowed
as it heated. Queued models (V1, V2) only, and it needs a zoompilot patch.

<a id="what-still-needs-validation"></a>

## Power and connection

Use separate power supplies for the comma and server. Voltage drops can reboot
the server and interrupt the link. Keep laptops powered and awake, and provide
adequate cooling during sustained use.

With always-on power, the Jetson sleeps when the ignition is off and the comma
wakes it when the car starts. See [power setup](transport.md#power-requirements)
for consumption and the separate battery-protection shutdown option.

TCP has no client authentication. Use a trusted network. Wi-Fi missed the frame
budget in measurements; use USB 3 or wired Ethernet.
