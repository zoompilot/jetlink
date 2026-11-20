# Add-ons

An add-on is a fork's own model served by the Jetson beside the driving model,
over the same USB cable: openpilot's sign vision or a fork's V-ASM, say. The
comma runs the model on its own CPU whenever the Jetson does not answer, so an
add-on only ever takes work off the comma.

Three parts, each in this repository:

- **The comma's interface.** The fork declares a `GadgetFunction` on its
  adapter, and Jetlink puts it on the gadget next to its own
  ([porting.md](porting.md#extra-usb-interfaces)).
- **The comma's client.** `jetlink.addons.link`: `AddonLink` owns the
  interface, and `JetsonNet` stands in for a `cv2.dnn` net, so a fork changes
  one line where it loads the net (`with_host`).
- **The server.** `jetlink-server --addons FILE` builds the add-on models on
  its own backend (TensorRT on a Jetson, onnxruntime on a Mac) and serves each
  add-on's interface.

From Peter Nguyen's V-ASM and sign-vision offload (trung791997/openpilot).

## The driving model first

The server runs an add-on pass only in the gap a driving frame leaves. Frames
come at a steady rate, so the next one is due a period after the last one
started. A pass runs when it ends, by its recent cost and a margin, before
then, and it holds the engine lock as a frame does, so the two never share the
GPU. A frame that comes early waits at most one pass. With no driving frame
for a second, nothing is driving and passes run at once.

A request may wait for a gap as long as the pass still ends inside the comma's
50 ms, which most often means through the end of the next frame. One that
finds none is answered `YIELD`, and the comma runs its own model for that
request alone. When three requests in two seconds find none, the hellos say
yielding until that passes, and the comma keeps to its own model until the
next plain hello.

Add-on models build after the driving model's own build or load, never beside
one, and stay resident: they are small.

## The comma

```python
from jetlink.addons.link import with_host
from jetlink.comma.functions import GadgetFunction

VASM = GadgetFunction('vasm', (0xFF, 0x56, 0x41))   # also on the adapter's gadget_functions()

net = cv2.dnn.readNetFromONNX(path)
net = with_host(net, VASM, sha256_of(path), [('output0', (1, 3))], scaled_u8=True, report=write_status)
```

`with_host` wraps the net in a `JetsonNet` on the process's one `AddonLink`
for that interface (`shared_link`), or returns the net itself if it cannot.
Wrap it whether or not the gadget carries the interface yet: the link looks
on each call, so a net loaded before the owner built the gadget still moves
once it has.

`outputs` names every graph output with its shape. The first is what
`forward()` with no names returns, as cv2 returns the graph's first; the
server answers them all in name order. `scaled_u8=True` promises a
`blobFromImage(img, 1/255)` blob and sends its u8 bytes, a quarter of the size
and exact. Leave it `False`, the default, for any other blob: it goes as f32.

`enabled` is a cheap callable read on every forward: a fork's setting turns
the host off and on without reloading the net. Declare the interface on the
gadget whatever the setting, as one nothing opens costs nothing, so a change
never needs the gadget rebuilt or the comma restarted.

`report(dict)` hears the link every 30 s while the caller asks or hellos
come: connected, yielding, the host's model tags, `missing` (the ones this
comma asks for that a connected host lacks, a different file on each end most
often), counts and round trips.

The client asks only while the link is live: a hello or an answer in the last
2.5 s, a hello that lists the model and does not say yielding, no write in
flight. It waits 50 ms for an answer.
A miss, a timeout or a bad answer, rests the link for 3 s, so a slow host
never holds the caller at the faster rate it keeps for the host.

It never opens an endpoint before a host has configured the gadget, and closes
both on any error. It reads in 16 KB pieces and writes in 8 KB ones, as
FunctionFS allocates one contiguous kernel buffer per call.

## The server

```json
{"addons": [
  {"name": "vasm", "class": "ff/56/41", "models": ["/opt/addons/v_asm_model.onnx"]},
  {"name": "slv", "class": "ff/53/4c", "models": ["/opt/addons/sign_detector.onnx", "/opt/addons/light_detector.onnx"]}
]}
```

`class` is the interface's class triple, as the fork's `GadgetFunction` has
it, and never Jetlink's `ff/ff/ff`. A relative model path is from the file's
directory. The server keeps the built artifacts in `<cache>/addons`, apart from
the driving models'.

A model takes one `[1, C, H, W]` input, float, fp16 or uint8, and gives any
number of float outputs. Its tag on the wire is the first four bytes of the
ONNX's sha256, so the comma and the server need the same file.

On a Jetson, copy the models over, write the file, and name it in
`/etc/jetlink/addons.env`, which the installer never writes:

```bash
sudo mkdir -p /opt/addons && sudo cp *.onnx addons.json /opt/addons/
echo 'JETLINK_ADDONS="--addons /opt/addons/addons.json"' | sudo tee /etc/jetlink/addons.env
sudo systemctl restart jetlink-server
```

`jetlink logs` shows each model built, then `serving 1 model(s) on interface
class ff/56/41` and `claimed its interface` once the comma's gadget carries it.
A minute's counts follow: passes, yields, failures and each model's cost.

On a Mac, `jetlink-server serve --usb --addons FILE`. The add-on models build
on onnxruntime's one-session CoreML profile, since the Neural Engine profile
splits a driving model in two.

## The wire

Version 1. Little-endian, one request and one answer at a time on each
interface, `jetlink/addons/wire.py` and `AddonWire.swift`.

| Message | Layout |
| --- | --- |
| request | `"JAQ1"` seq:u32 tag:u32 c:u16 h:u16 w:u16 encoding:u8 version:u8 nbytes:u32, then the image |
| answer | `"JAR1"` seq:u32 tag:u32 status:i32 server_us:u32 count:u32, then count f32 |
| hello | an answer with seq `0xFFFFFFFF`, tag the version, status `HELLO` or `YIELD`, then count u32 model tags |

Encoding 0 is u8, which the server divides by 255; 1 is f32. Statuses: `OK` 0,
`BAD_REQUEST` 1, `INFER_FAILED` 2, `UNKNOWN_MODEL` 3, `HELLO` 100, `YIELD` 101.
`server_us` is the server's time from the request to its answer.

A message whose length is a multiple of 512 carries one zero byte more, so it
always ends in a short packet: FunctionFS reads end at one, and the host
sends no zero-length packets. A message is at most 4 MiB.

The server sends a hello after a second with no request. A malformed request
ends the link, since a byte stream has no next message to skip to; the server
claims the interface again.
