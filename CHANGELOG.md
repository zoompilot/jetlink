<!-- Each release: a "Jetlink vX.Y.Z" heading underlined with =, newest first. A release that changes the fork contract ends with a **Forks** group (docs/publishing.md). -->
Jetlink v0.9.0
==============
**Models**
* The big-model list is jetlink-model-index's: every big model comma has published, each ONNX listed once in sunnypilot's folders, and open comma pull requests in their own folder. It is checked every hour, so a new model shows up without a release, and a model sunnypilot rebuilt keeps its pick. The comma and the Jetson each keep the last list they fetched, so either one can pick offline. **Update the Jetson and the comma.**
* The default big model is the index's too, so it can change without a release.
* The Jetson and the comma download from comma's new HuggingFace storage first.

**Forks**
* API 4; API 3 adapters keep working until the next minor release. New: `lists_big_models()`, `big_catalog()` (the index as a catalog in sunnypilot's shape, or None offline) and `listed_ref(ref)`, which replace `should_extend_catalog()` and `extend_catalog()`. `Keys.pointers` (the `JetlinkModelPointers` param) is no longer read and can go from params_keys.h. Removed: `catalog/extra_big_models.json` and the comma's own lookups of a commit's ONNX; the comma keeps the index as `index.json` in its model directory.

Jetlink v0.8.13
===============
**Models**
* Fixed a big model picked on the comma going back to the one picked before. Choosing the Default reverted every time, and after the Jetson restarted with its clock behind (offline in the car), any pick could. **Update the Jetson and the comma.**

**Forks**
* API stays 3. An emptied big-model slot that held a pick now counts as choosing jetlink's default, which goes to the server like any pick; a slot never filled still takes the server's pick. jetlink keeps the slot's history in `slot.json` under its model directory.

Jetlink v0.8.12
===============
**Models**
* A fork whose model list knows a model only by its ONNX file, such as StarPilot, can pick the big model on the comma, and its list shows a model picked in the Jetlink app.

**General Updates & Fixes**
* Fixed the Jetson's web page, which did not load in v0.8.11. **Update the Jetson.**

**Forks**
* API stays 3. New: the big-model pick may name its model by the ONNX's hash, `{sha256, displayName}`; jetlink sizes it from comma's LFS servers. With `Keys.big_model` and no catalog, a newer pick made on the server is written into that param.

Jetlink v0.8.11
===============
**Models**
* Allow model selection from the comma or the Jetlink app without redownloading the model on each device.

**General Updates & Fixes**
* **Update the comma and the app:** either can go first. zoompilot's `danger-unstable` branch has the comma's side.

**Forks**
* API stays 3. New: the optional `Keys.select`, the param jetlink writes a ref to when a newer pick was made on the server (sunnypilot's `ModelManager_DownloadRef`). Without it, or without a model picker, the server's pick still drives.

Jetlink v0.8.10
===============
**General Updates & Fixes**
* Fixed high CPU use on a Mac, iPhone or iPad while a comma is connected but not driving.

**Forks**
* API stays 3. New: the optional `ModelSide.model_warp()`, for a fork whose runtime cannot load openpilot's warp; move its output with `warp.make_coherent` before the JIT first runs.

Jetlink v0.8.9
==============
**Models**
* Fixed Mountain Dew Model v3 and Flip Flop Model showing as one.

Jetlink v0.8.8
==============
**Models**
* **Flip Flop Model** and **Mountain Dew Model v3** are listed separately. Picking one no longer picks the other.

**Comma**
* A comma 3X no longer powers off when a laptop is plugged into its port.
* Skips downloading a model the Mac or Jetson on the cable already has.

**General Updates & Fixes**
* A big model that returns NaN hands back to the small model, keeping the link.
* Stays ready while the comma is connected, so rejoining mid-drive is fast. Thanks Danny (@dirwin31).
* **Update the app.** The comma gets v0.8.8 with zoompilot's next `main` release.

**Forks**
* API stays 3. New: `Hooks.runtime()` (modeld's runtime record) and `Hooks.swap_events()` (selfdrived's swap events, built once and given an in-control bool each tick; they read the runtime record themselves unless given the fork's own state), `Jetlink.runtime()`, and `FakeSubMaster` and `check_fallback()` in `jetlink.openpilot.testing`. With `link_on()` saying off, only the hooks in `BINDS_WITH_LINK_OFF` bind (status, shutdown and catalog); the rest, `prepare()` included, answer as the link off. A fork that calls these must pin this release or newer: an older checkout has no such hooks.
* API history: 2 (2026-10-05) had modeld write in_control onto the joining model; 3 (2026-10-10) has the joining model ask the adapter's in_control() itself. Jetlink.shutdown() and BuildSide.make_warp went at API 2 without a bump.

Jetlink v0.8.7
==============
**Models**
* **Mountain Dew Model** now runs the retrained model comma shipped, not the earlier one.
* **Mountain Dew V2 Preview** removed. If you picked it, Jetlink switches to Cinque Terre V3; pick **Mountain Dew Model** for comma's final version.
* **Mountain Dew Model v3** is listed, and loads once comma publishes its files.

**General Updates & Fixes**
* **Update the comma and the app:** zoompilot's `develop` branch pins Jetlink v0.8.7.

Jetlink v0.8.6
==============
**Jetson**
* Sleeps until the comma wakes it, no longer waking every 30 minutes.

**Models**
* **Mountain Dew V2 Preview** added.

**General Updates & Fixes**
* Works with the latest openpilot.
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.6.

Jetlink v0.8.5
==============
**iPhone & iPad**
* Faster, steadier cable link.

**Driving**
* One slow frame no longer hands the big model back.
* Fixed switching to an old model while the pick builds.

**General Updates & Fixes**
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.5.

Jetlink v0.8.4
==============
**Driving**
* A slow big model hands back within a second.
* modeld uses less CPU.
* Fixed occasional link drops while driving.

**iPhone & iPad**
* The comma no longer charges the phone. It caused instability.
* Status says **Driving on Small Model** after a lost link.

**Models**
* **ResAction Preview** added.
* The last built model drives while a new pick downloads.

**General Updates & Fixes**
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.4.

Jetlink v0.8.3
==============
**iPhone & iPad**
* **USB-C to C Cable:** Works without a hub, and the comma charges the phone.
* Frames reach the phone faster over the cable.

**General Updates & Fixes**
* The Neural Engine stays warm between sessions, so the first frame after a rejoin is on time (iPhone, iPad, Mac).
* The comma 4 no longer gets stuck on "waiting for jetlink" after a rejoin.
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.3.

Jetlink v0.8.2
==============
**Mac**
* **Updates:** The app now updates itself. From 0.8.1, replace it by hand one last time.
* **Lid Closed:** New setting to keep serving with the lid closed. Thanks Nick (@mzdnick)!

**General Updates & Fixes**
* The comma's setting is now called **Jetlink** (was Accelerator Link).
* The comma 4 says to check the cable after repeated drops.
* Removed TCP debugging from the UI.
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.2.

Jetlink v0.8.1
==============
**Driving**
* The big model warms up while the small model drives, so the switch is smooth.
* A late frame reuses the last plan instead of dropping. Five in a row, or 20 in 10 s, still hand back to small model.
* A big model that falls behind rejoins in about a second (was up to 60 s).

**iPhone & iPad**
* A banner on the Status tab shows when the link is USB 2, and why.
* Logs say why the comma stopped using the big model.
* **USBC to C Cable:** work in progress, try a USBC->USBA cable with USBA->C adaptor, flipping usb cable, or high quality hub.

**Android**
* Download the APK from the release page. No build needed <3
* **Automatic:** The new default. Uses a Pixel's NPU, else the GPU.
* Google has not allowed Jetlink on the Pixel NPU yet, so Pixels use the GPU for now.

**General Updates & Fixes**
* Turning on the Accelerator Link turns ADB off. They share the USB port.
* **Update the comma:** zoompilot's `develop` branch pins Jetlink v0.8.1.

Jetlink v0.8.0
==============
**iPhone & iPad on TestFlight!**
* Install from [TestFlight](https://testflight.apple.com/join/DAsYk5sP). No Xcode needed <3
* **Tested:** Driven with an iPhone in the car.
* **Direct Cable:** USB-C to USB-C should now work without a hub. Not yet tested.

**Android on more phones**
* Big models run on any phone's GPU, not just Snapdragon. The GPU is now the default.
* **NPU:** Still an option on Snapdragon.
* Not yet tested on a phone. Run the Benchmark before you drive.

**Driving**
* **Slow Model Handback:** A throttling iPhone no longer forces a disengagement. The small model takes over instead.

**General Updates & Fixes**
* **Faster Jetson Start:** Ready in 25 s (was 38.5). Run `jetlink update`.
* **Jetson Desktop:** The installer can turn it off to free memory. Existing installs: `jetlink setup`.
* **Installer:** Shorter, plainer questions.
* **Benchmark:** A run with no frames now says **Too Slow**.
* **Loading:** Shows seconds instead of a stuck 0%.
* **Cable:** Use a USB 3 USB-C cable for the Mac, iPhone and Android.
* **Update the comma:** zoompilot's `jetson-trt` branch pins Jetlink v0.8.0.

Jetlink v0.7.4
==============
**General Updates & Fixes**
* **Linux PCs:** The installer runs on Debian 12, Fedora, Arch and openSUSE as well as Ubuntu: the base packages come from apt, dnf, pacman or zypper, the NVIDIA driver is installed on Arch and on Ubuntu's and Arch's derivatives too (printed as the distribution's own steps elsewhere), and a system without systemd or with a glibc older than Ubuntu 22.04's is refused up front. Ubuntu is what users have tested; the others are untested on hardware.

Jetlink v0.7.3
==============
**General Updates & Fixes**
* **No False LOW MEMORY Alert:** With the Accelerator Link on, the comma counted about 10% more memory as used than it was, and drives at a real 80% showed **LOW MEMORY** with a takeover warning. The free-memory floor Jetlink set on the comma is gone; the dirty-memory caps that keep the link smooth stay. Driven with no regression.
* **Update the comma:** zoompilot's `develop`, `danger-unstable` and `jetson-trt` branches pin Jetlink v0.7.3. All of this runs on the comma; the server needs no update.

Jetlink v0.7.2
==============
**General Updates & Fixes**
* **Cinque Terre V3 in sunnylink:** sunnylink's model selector now lists Cinque Terre V3 (zoompilot reads sunnypilot's newer big-model list).
* **Model List:** A spotty connection no longer drops a newer model from the list. It could reset your pick and quietly switch the default to Cinque Terre V2.
* **Update the comma:** zoompilot's `jetson-trt` branch pins Jetlink v0.7.2. All of this runs on the comma; the server needs no update.

Jetlink v0.7.1
==============
**Driving**
* **Stays Engaged:** If the big model drops or falls behind while engaged, the comma shows **TAKE CONTROL** for 5 seconds and keeps driving on the small model (was a soft disable). Lateral-only (MADS) driving gets the same warning.
* **No Lag Alert After a Pull:** The small model takes over within a frame, the first pull of a drive included (was over a second), with no "Driving Model Lagging" afterwards. A server that stops answering is caught in 0.2 s (was 0.5).
* **Switch Without Stopping:** Connected mid-drive? Turn cruise fully off and engage again: the next engagement uses the big model. The comma says **Big Model Ready: Re-engage to switch** once, not at every stop, and **Big Model Active** when you can engage on it.
* **Steering:** No kick when the big model hands back mid-drive.
* **Replug:** A Jetson plugged back in reconnects at once (could wait up to a minute).

**General Updates & Fixes**
* **Accelerator Link:** Switching out of iOS no longer needs a power cycle: change it with the car off.
* **Mac:** The Models inspector holds the model's actions; the toolbar button only opens it.
* **Update the comma:** zoompilot's `jetson-trt` branch pins Jetlink v0.7.1. All of this runs on the comma; the server needs no update.

Jetlink v0.7.0
==============
**Android support!**
* Run big models on a Snapdragon phone's NPU over USB (experimental, build from source) <3
* Turn on: Settings > Models > Accelerator Link > USB.

**One Swift engine everywhere**
* Jetsons, Linux PCs, the Mac, iPhone/iPad and Android all run the same Swift server.
* On a Jetson it gives v0.6.0's output bit for bit, with half the CPU.
* **No Docker:** Jetsons and PCs run it natively. `jetlink update` moves your install over, keeping settings, models and engines.
* **Update the comma and Jetlink together:** New link protocol; zoompilot's `jetson-trt` branch pins Jetlink v0.7.0. A comma on an older build stays on the small model.

**General Updates & Fixes**
* **Faster Link:** Big-model frames 3 to 5 ms quicker on the bench Jetson: 2.2 ms of USB transport (was 8.5). Replies are 8 KB (was 74 KB): the hidden state stays on the server. USB 3 power saving is off only while the comma is sending.
* **Parked:** The Jetson sleeps between its half-hourly wake checks: awake about 1% of a park (was 6%).
* **Reliability:** The comma restarts Jetlink if it stops, alerts when it cannot, and no longer stalls at shutdown. A new model the Jetson has never seen uploads right away. "Power off with the comma" reaches the Jetson every time.
* **Forks:** openpilot forks plug Jetlink in through one adapter module (developers: `jetlink/openpilot`).
* **Status Page:** Watch the server from your phone at `http://<name>.local:5600`.
* **Stay Awake:** `jetlink caffeinate` keeps a Jetson awake while you work on it.
* **iPhone:** Neural Engine + GPU is the default (14 ms a frame on an iPhone 18 Pro).
* **Linux PCs:** The installer puts NVIDIA's TensorRT 11.3 in `/opt/jetlink`; no system TensorRT.
* **Tested:** JetPack 7.2.1 and the Mac, in the car. JetPack 6.2, Linux PCs and WSL2 are untested.

**Removed**
* Python server, Docker images and `scripts/run-mac.sh` (use `jetlink-server`).

Jetlink v0.6.0
==============
**Mac**
* **Swift only:** No bundled Python; 14 MB download (was 120 MB).
* **Settings:** Server and Log level removed.
* The Python server still runs from a checkout: `scripts/run-mac.sh`.

Jetlink v0.5.0
==============
**iPhone & iPad support!**
* Run big models on an iPhone or iPad over one USB cable (experimental, build with Xcode) <3
* Turn on: Settings > Models > Accelerator Link > iOS.
* Thank you Casey (@ScriptDrifter) for the original iPhone port!

**One Swift engine**
* The Mac and iPhone/iPad apps now share one Swift server, matched to the Python server's output.

**General Updates & Fixes**
* **Accelerator Link:** Now Off, USB or iOS. Set it again after updating.
* **Default Model:** Cinque Terre V3.
* **Model Switching:** New models download and build in one step.
* **Reliability:** Recovers in seconds if the Jetson server restarts mid-drive.
* **Jetson Power Off:** Fixed the comma not shutting the Jetson down.
* **Installer:** Installs the latest release; `jetlink update` moves to the newest.
* **Mac:** Swift server option with no Python, and a Benchmark page.

**Removed**
* tinygrad backend and Jetson over Ethernet.

Jetlink v0.4.3
==============
**Mac**
* **Fixed:** "No bundled Python runtime" when opening from Finder or the Dock on macOS 15.

Jetlink v0.4.2
==============
**Mac**
* **Signed & Notarized:** Opens without the Gatekeeper workaround.

Jetlink v0.4.0
==============
**One-command install!**
* Sets up your Jetson or Linux PC with one command.
* `jetlink update` updates, and rolls back if it fails.

**General Updates & Fixes**
* **JetPack:** 6.2 and 7.2 (7.2 is ~12% faster).
* **Linux PCs:** NVIDIA driver 580+.
* **Models:** Cinque Terre V3, and new big models show up without a Jetlink update.
* **Jetson Sleep:** Fixed on stock JetPack.

**Mac**
* **Apple Silicon:** M1 Pro and up, with the model split across the Neural Engine and GPU.
* **Models:** Preload with one click. Models from v0.3.0a1 prepare again once.
* **Latency:** New visualizations.
* **Branding:** New Jetlink look and icon.
