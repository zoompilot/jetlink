# Link protocol and USB reference

For cable and power setup, see [Cables and power](transport.md).

## Link protocol

The comma (`jetlink/protocol.py` in this repo) and every server speak protocol
3, over the vendor interface's bulk pipes or over TCP (a phone's cable network,
bench tools). There is one version: update the comma and Jetlink together.

- **Messages.** A 32-byte header (magic `JLNK`, version 3, type, sequence
  number, flags, length) and a payload. Every message carries version 3. A
  header with any other version is a broken stream: the server drops the
  link, a comma on another version gets no answer to its hello, and it drives
  on its small model.
- **Picks.** The comma's hello carries its big-model pick, how long ago it
  was made and when by the comma's clock (`pick`: ref, sha256, nbytes, name,
  `age_s`, `at`); the default, chosen by emptying the slot, is a pick too. The
  server keeps its own pick (Use Model, the web page's prepare) beside the
  cache (`pick.json`) with the pick the comma named last, and its reply's
  `pick` is the newer of the two, with its age; the comma then runs that one
  and asks its model manager for it, unless it has picked again since the
  hello. Newer is settled by the comma's last hello where it can be: a pick
  the comma has named loses to the next one it names, and one made on the
  server wins while the comma still names the pick it named before, whatever
  either clock did (a Jetson's starts again behind at every power-on). Only
  when both ends picked since the last hello do the dates decide, each by its
  own clock with ages crossing the link. The reply's
  `fetches` says whether the server downloads a model itself when asked: a
  default route and no download failed for want of the internet in the last
  10 minutes.
- **Cameras.** Once the comma is calibrated, its hello also carries what
  calibrationd saved of its cameras (`camera`: `wide_from_device_euler`, the
  wide road camera's roll, pitch and yaw from the device, and `height`, the
  camera's height above the road), read once a link, for whatever on the
  server needs to place what the cameras saw. A server that predates it
  ignores the key.
- **Session.** The comma says hello, asks for its model's engine (with
  `fetch` when it has not got the model either: the server then downloads it,
  state `downloading`, stage `download`, and builds it; a server that cannot
  answers `need_upload` and the comma downloads and uploads it), waits until it is ready, then sends one
  INFER_REQ per camera frame, 20 a second, from then on: whether the big
  model is driving or the small one still is, so the big model's history and
  hidden state are current and warm when it takes over. The server cannot
  tell the two apart. Until the first frame goes out it pings; a server that
  has not heard this comma's hello (it restarted meanwhile) answers with an
  error, and the comma says hello again.
- **Late replies.** With the big model driving, a reply not back 46 ms into
  the frame is not waited for: the comma publishes the previous frame's
  output again and reads the late reply with the next frame's. Five of those
  in a row, or more than twenty in ten seconds, hand the drive back to the
  small model (`behind` in the leave). A host that answers nothing for 0.2 s
  has lost the link, whichever model is driving (`lost`).
- **INFER_REQ.** The comma's warped camera images (uint8) and 12 floats
  (`desire`, `traffic_convention`, `action_t`): 393,304 bytes with the header,
  409,600 over USB with the comma's padding.
- **Frames over a phone's cable.** A phone's server offers a UDP port and a
  token in its hello reply, over the cable only. Once the big model has an
  output to fall back on, the comma sends each INFER_REQ there as about seven
  equal UDP datagrams (`JFRM` header: token, sequence number, offset, total)
  instead of on the TCP connection. The comma's kernel then does its
  per-packet work about seven times a frame instead of once for each of
  ~270 TCP segments: 7.35 to 4.3 ms a frame on the bench. Everything else, the
  replies included, stays on TCP. A frame missing a piece is dropped when the
  next one starts; the comma never gets its reply and treats it as a late
  reply.
- **INFER_RESP.** The status, the server's timings, and the model's outputs in
  float32 without `hidden_state`: 8,324 bytes for the current models (73,860
  with it). The comma sets WANT_HIDDEN on a frame to get the whole vector, for
  logging every output; WANT_STATE appends the server's telemetry as JSON.
- **Hidden state on the server.** A queued model (BMRLNAP, Cinque Terre V2,
  Lebowski) feeds each frame's `hidden_state` into the next frame, where
  openpilot's modeld feeds its own. The server keeps it: zero when an engine
  loads, on every hello and on a frame flagged RESET_QUEUES, and replaced only
  after a frame whose outputs are all finite (a NOT_FINITE or failed frame
  leaves the last good one). Outputs are bit for bit what they were when the
  comma sent the state back each frame. A stateful model (Cinque Terre V3)
  keeps its state in the engine.
- **Padding.** A bulk transfer ends on a short packet. Over USB the comma pads
  every message it sends to a multiple of 16 KB, so none ends on a short
  packet. Everything else (messages to the comma, and TCP both ways) gets one
  pad byte when a message's length is an exact multiple of 512 bytes (the
  USB 2 packet, which divides USB 3's), so a client speaks to every transport
  the same way and a link that fell back to USB 2 still ends every message.
- **Host reads.** Every USB host (usbfs on Linux and Android, IOKit on a Mac)
  keeps 32 reads of 16 KB posted on the comma's pipe, so a whole request
  streams without the server asking for the next piece, and hands the bytes
  over in the order the reads were posted. Since each message ends on the
  16 KB grid, a read never holds the end of one message while it waits for
  the next. A short packet means the stream left the grid, which only a
  broken stream does: the session ends and the comma reconnects.
- **Link power.** On Linux, USB 3 link power management is off on the comma's
  port while a session is served and the comma has sent something in the last
  30 s ([USB speed and power](#bus-speed)).

Measured on the bench Jetson: [performance](status.md#measured-performance).

### What the comma presents

One of two USB gadgets, per the comma's **Jetlink** setting (models
settings):

| Setting | For | Gadget |
| --- | --- | --- |
| **USB** | Jetson, Mac, Linux PC, Android | Plain: one vendor-specific interface, one bulk endpoint pair, opened through usbfs on Linux (IOKit on the Mac; usbfs on the descriptor Android's USB host API hands the app). No network interface. |
| **iOS** | iPhone | Composite: interface 0 is the same vendor interface (never used on iOS), then a CDC-NCM network interface, since iOS gives apps no vendor USB access but drives USB network adapters itself. |

- iOS network: the comma is `192.168.60.1` and runs DHCP; the phone gets a
  `192.168.60.x` address with no gateway or DNS, keeps its internet route over
  Wi-Fi, and dials `192.168.60.1:5599`.
- Changing the setting rebuilds the gadget (an unplug), so it changes only
  offroad.
- Comma side: the `jetlink.comma` package. The owner holds the gadget and lends
  modeld its endpoints, or on the cable the right to take the phone's dial:
  whoever holds the loan listens for it, the owner while nobody does. Every
  root step goes through `jetlink/comma/jetlink-root.sh`. See the
  [installation reference](installation-reference.md#custom-usb-integrations).

### Bus speed

Latency needs USB 3 (SuperSpeed). A frame is about 400 KB to the server and
8 KB back (the model's hidden state stays on the server): about 1 ms on USB 3,
10 ms on USB 2 (hence USB 3 on every hop: cable, adapter, any hub).

On Linux the server turns off USB 3 link power management on the comma's port
while it serves the comma, and puts the kernel's default back when the session
ends or the comma has sent nothing for 30 s (parked with its gadget still up),
until its next message: waking the link from its low-power states cost 2.2 ms a
frame on the bench Jetson, and keeping it awake with nothing to carry costs
0.18 W ([details](installation-reference.md#custom-usb-integrations)).

Negotiated speed on the comma: `/sys/class/udc/*/current_speed`
(`super-speed` is USB 3, `high-speed` USB 2), also printed with the built
gadget by `sudo jetlink/comma/jetlink-root.sh check`.

### The network link on a Linux host

The comma's kernel (4.9, Qualcomm's u_ether) sends NCM blocks slowly when the
host lets it pack several packets into one. Comma to host, bench mici,
SuperSpeed, Jetson host:

| NCM block size | Throughput |
| --- | ---: |
| 16 KB (Linux default) | 22 Mbit/s |
| 2 KB | 190 Mbit/s |

Host to comma is unaffected (340 Mbit/s); CPU is not the limit. A Linux host
using the network link (a bench standing in for a phone, or a PC over the cable
network) should cap the block size:

    echo 2048 > /sys/class/net/<interface>/cdc_ncm/rx_max

`scripts/99-jetlink-host.rules` does that on plug-in. Apple's NCM driver picks
its own block size and showed no slow path (reference phone: 393 KB up in under
19 ms).

With the cap, parked live bench on the comma (Cinque Terre V3, 180 s, 3,416 big
frames, every frame delivered; 2026-09-27, with the earlier link protocol's
74 KB replies):

| Link | p50 | p99 |
| --- | ---: | ---: |
| Network link | 36.4 ms | 40.6 ms |
| Vendor interface | 28.6 ms | 30.3 ms |

The ~8 ms gap is all in the comma's send of the 393 KB frame (`bench_link.py`:
26 ms of transport vs 8.6 ms). That is the network link's floor on this kernel;
the vendor interface stays the link for every host that can open it.
