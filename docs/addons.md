# Add-ons

An add-on is a fork's own model served by the Jetson beside the driving model,
over the same USB cable: openpilot's sign vision or a fork's V-ASM, say. The
comma runs the model on its own CPU whenever the Jetson does not answer, so an
add-on only ever takes work off the comma.

From Peter Nguyen's V-ASM and sign-vision offload (trung791997/openpilot).

## Offload a model

Three changes in a fork, and nothing on the Jetson: `jetlink-server --usb`
(and the Mac app over USB) serves any add-on a comma provisions.

**1. Check the model.** One input, `[1, C, H, W]`, float, fp16 or uint8, and
any number of float outputs. The comma names it on the wire by its ONNX's
sha256, so pin the hash of the very file the comma runs.

**2. Declare the add-on** in the adapter module, inside `_adapter_class()` like
the rest, since its top level imports the standard library only:

```python
def _adapter_class():
  from jetlink.openpilot.stock import Stock

  class Adapter(Stock):
    ADDONS = (vasm(),)
  return Adapter


def vasm():
  from jetlink.addons.addon import Addon
  return Addon('vasm', (0xFF, 0x56, 0x41), setting='MyForkVasmOnJetson',
               models=(('V-ASM', BASEDIR / 'models/v_asm_model.onnx', V_ASM_SHA256),))
```

- `name`: 1 to 16 of `a-z0-9`, the gadget function's name.
- `iface_class`: a vendor class triple of your own, never Jetlink's
  `(0xFF, 0xFF, 0xFF)` or another add-on's.
- `setting`: a BOOL param that turns it on, or None for always. Declare it in
  the fork's params keys. Its changes start a provisioning run, and the model
  process reads it live.
- `models`: `(name, path, sha256)` for each model the add-on serves. `name`
  is what the log and the progress show.

`ADDONS` puts each interface on the gadget whatever its setting (one nothing
opens costs nothing), sends a Jetson that lacks them the models of those turned
on, and watches their settings. A fork that needs more overrides
`gadget_functions` or `addon_models` on its adapter instead
([porting.md](porting.md#extra-usb-interfaces)).

**3. Wrap the net** where the fork loads it:

```python
net = cv2.dnn.readNetFromONNX(path)
net = vasm().wrap(net, V_ASM_SHA256, [('output0', (1, 3))], params=Params(), scaled_u8=True)
```

`outputs` names every graph output with its shape, the one `forward()` returns
first. `scaled_u8=True` promises a `blobFromImage(img, 1/255)` blob, sent as
its u8 bytes; leave it off for any other blob. The wrapper asks the Jetson
while the setting is on and the Jetson has the model, and runs the net itself
otherwise; it returns the net itself if it cannot be made.

Any other runtime (onnxruntime, tinygrad) wraps its own run as a call that
returns every output in `outputs` order:

```python
run = vasm().model(lambda blob: session.run(['output0'], {'images': blob}), V_ASM_SHA256, [('output0', (1, 3))],
                   params=Params(), scaled_u8=True)
(scores,) = run(blob)
```

**4. A panel (optional).** `Addon.present()` (the gadget carries it),
`Addon.report()` (the link's last report: connected, yielding, counts, round
trips) and `Addon.state(server, progress)`, from the owner's record of the
server's hello and the fork's progress param:

```python
from jetlink.comma import gadget
state = vasm().state(gadget.read_record(gadget.SERVER), params.get('JetlinkProgress'))
# {'jetson': 'ready' | 'missing' | 'failed' | 'unsupported' | None, 'preparing': 'V-ASM: build 40%' or ''}
```

`missing` prepares the next time the car is parked, `failed` is a model this
Jetson cannot build or run, and `unsupported` a Jetson whose JetLink is too
old for add-ons.

**Then check it**: turn the setting on while parked. The progress shows the
upload and the build, the Jetson sleeps after, and on the next drive the
report says connected with answers counted.

## How it fits together

- **The comma's interface.** The add-on's `GadgetFunction`, which Jetlink puts
  on the gadget next to its own.
- **The comma's client.** `jetlink.addons.link`: `AddonLink` owns the
  interface, `HostModel` wraps any runtime's model as a call (`host_model`,
  which `Addon.model` calls), and `JetsonNet` stands in for a `cv2.dnn` net
  (`with_host`, which `Addon.wrap` calls).
- **Provisioning.** The parked provisioning run sends the Jetson the models it
  lacks (`Stock.addon_models`), over the main link, after the driving model.
- **The server.** `jetlink-server --usb` builds the add-on models on its own
  backend (TensorRT on a Jetson, onnxruntime on a Mac) and serves each
  add-on's interface. The Mac app does the same over USB.

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

`Addon.wrap` calls `with_host`, which wraps the net in a `JetsonNet` (and
`Addon.model` calls `host_model`, a `HostModel`) on the process's one
`AddonLink` for that interface (`shared_link`), or returns the net itself if it
cannot.
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

## Provisioning

A server that serves add-ons says so in its hello: `addons`, `{ready, failed}`
by sha256. A parked run sends it each add-on model in neither, once the
driving model is ready: an `ENGINE_REQ` with `addon: {interface, name}`, then
the usual upload, `PROGRESS` (progress codes `addon_upload` and `addon_build`)
and `ENGINE_RESP`, whose `ready` carries no spec. The driving engine stays
loaded. A server too old to list `addons` is never sent one, as it would take
it for its driving model.

Then the run says hello again, so the owner's record (`gadget.SERVER`) lists
them, and the Jetson sleeps as after any run. A model the server cannot build
or run is listed `failed` until its backend changes, so it is not sent on
every park; deleting `<cache>/addons/failed.json` on the Jetson tries again.
`Addon.state` reads it for a panel.

## The server

With `--usb` the server serves every add-on a comma provisions, from its cache
after a restart too: `<cache>/addons`, a cache of its own beside the driving
models', which keeps the newest 8 artifacts per backend and the interface each
serves. `--no-addons` leaves the comma's add-on interfaces to another program.

A model takes one `[1, C, H, W]` input, float, fp16 or uint8, and gives any
number of float outputs. Its tag on the wire is the first four bytes of the
ONNX's sha256, so the comma and the server need the same file.

`jetlink logs` shows each model built, then `serving interface class ff/56/41`
and `claimed its interface` once the comma's gadget carries it.
A minute's counts follow: passes, yields, failures and each model's cost.

On a Mac, `jetlink-server serve --usb`. The add-on models build
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

The server sends a hello after a second with no request, and every 2 s
regardless, with the models ready then. A malformed request
ends the link, since a byte stream has no next message to skip to; the server
claims the interface again.
