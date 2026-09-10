# 60. Documentation

**Owner: agent G.** Files you create: `docs/macos-app.md`, `docs/models.md`.
Files you edit: `README.md`, `docs/platforms.md`, `docs/jetson.md`,
`docs/backends.md` (one line). Read first: `00-architecture.md`, the existing
`README.md` and `docs/*.md` so the voice matches: short sentences, tables for
choices, a "what to expect" section, no marketing. Write for a user who owns a
comma and a Mac and has never opened a terminal; keep terminal material in the
developer sections.

## `docs/macos-app.md` (the user guide)

Sections, in order:

1. **What it does.** Three sentences: runs the server, keeps the right model
   loaded, manages downloads. One screenshot placeholder per screen
   (`docs/assets/macos-*.png`, to be captured by the integrator; reference
   them, do not fabricate).
2. **Requirements.** Apple silicon Mac; macOS 15 or later; 16 GB memory
   recommended (preparing a CoreML model peaks near 8 GB); about 7 GB of disk
   per model (1 GB download plus 5.5 GB prepared engine); a USB-A port on a
   hub, dock or adapter and a USB 3 A-to-C data cable.
3. **Install.** Download the DMG from the Releases page, drag to
   Applications, open. First launch: the server starts, the Status screen says
   "Waiting for comma". For a build marked unsigned (pre-release or a fork):
   the exact Gatekeeper sentence and the fix (right-click, Open; or
   `xattr -d com.apple.quarantine /Applications/Jetlink.app`).
4. **Prepare a model before you drive.** Open Models; the list is the same
   one as Settings > Models > Big Model on the comma. Pick the one the comma
   uses (the "Default" tag marks the fork's default); Download; Prepare. What
   the progress means; that CoreML takes about 9 minutes; that "Loaded" means
   the comma will be ready at once. Keep Jetlink running: closing the window
   is fine, quitting stops the server.
5. **Plug in.** Same as the README's step 3, plus: the Status screen shows
   "Connected over USB", frames per second, and frame time; what "Slow
   frames" means.
6. **Settings.** Each setting in one line, from `30-swiftui-views.md`. The
   cache folder and how to point it at an existing `models_cache/` from
   `scripts/run-mac.sh`.
7. **Backends.** The three choices with the M1 Pro numbers from
   `docs/backends.md` (link to it), and "Automatic is CoreML on the GPU".
8. **Troubleshooting.** A table: server failed to start (open Logs; the last
   lines say why; the usual cause is another server holding the USB device or
   a cache folder that is not writable); waiting forever (USB-A, data cable,
   the comma's Accelerator Link toggle); preparing takes long (9 minutes is
   normal; memory pressure makes it longer); "Big Model Lost" on the comma
   (cable, sleep, battery: see the keep-awake setting); the app was updated
   and the model rebuilt (a runtime change re-prepares; the download is kept).
9. **Where things live.** The cache folder, the log file, and how to
   uninstall (delete the app, the Application Support folder, the Logs folder;
   remove the login item).
10. **For developers.** Two lines pointing at `macos/README.md` and
    `docs/models.md`.

## `docs/models.md` (the CLI, every platform)

`jetlink-models` from `01-contracts.md` section 3.2, with one example per
subcommand and its output shape. Explain: what a ref is (a comma commit),
what the SHA-256 is (the file the comma asks for), where files go
(`<cache>/models`, `<cache>/engines`, `<cache>/registry`), and the two ways to
get a model onto a server: let the comma upload it (default), or prefetch
with `jetlink-models fetch <ref>` and prepare with `jetlink-models prepare
<ref>` or `jetlink-server --build`. State the rule: do not run `prepare`
while a `jetlink-server` is using the same cache; on the Jetson, stop the
service first, or use the control channel.

Then the control channel for integrators: `--control-socket`, the on-connect
events, one worked example with `socat` or `nc -U`, and a pointer to the
protocol table (copy the events and commands tables from `01-contracts.md`
section 4 into this doc; the plan files are not user documentation and may be
deleted).

Jetson specifics: inside Docker,
`sudo docker run --rm -it -v /mnt/data/jetlink:/mnt/data/jetlink --entrypoint python3 jetlink:latest -m jetlink.registry fetch <ref>`,
and that the image published to GHCR carries the same command.

## Edits

- `README.md`: in "2. Start the server", the Mac path becomes "Download
  Jetlink for Mac from the Releases page, open it, and leave it running. See
  the [Mac guide](docs/macos-app.md). Developers can still use
  `scripts/run-mac.sh`." Keep the rest. In "More", add the two new docs.
- `docs/platforms.md`: the Mac section starts with the app (two sentences and
  a link), then keeps the terminal instructions under a "From a terminal"
  subheading. Add a "Prefetching models" paragraph pointing at `docs/models.md`
  in the Linux section.
- `docs/jetson.md`: a short "Prefetching a model" subsection with the Docker
  command above, and a note that the control socket is available with
  `--control-socket` if someone wants to script the server.
- `docs/backends.md`: under "Two things the runtimes made the server do
  differently", add the third: "tinygrad must come from git at or after
  `e837e367aac9`; the PyPI 0.14.0 wheel has no `org.tinygrad` ONNX domain and
  cannot load the exported models." (Agent A or the integrator fixes
  `scripts/run-mac.sh`'s install line accordingly: replace the `tinygrad`
  extra with the git pin; note it in your report if it is still unfixed.)

## Style checks before reporting

No em dashes: `grep -rn $(printf "\xe2\x80\x94") docs README.md` must return nothing. Every relative
link resolves (`for f in docs/*.md; do grep -o '](\.\./[^)]*\|]([a-z./#-]*' ...` or just click through). Sentences under 25 words where possible. Tables
for anything with three or more parallel items.
