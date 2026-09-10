# Updates and rollback

The comma build and the Jetlink server have to match. Update both while parked.

## Which Jetlink to run

Clone `main`. The zoompilot fork records the exact Jetlink commit it was tested
with as its `jetlink_repo` submodule, and `main` is kept compatible with the
current `jetson-trt` branch. A protocol mismatch is refused cleanly: the server
rejects the connection and the comma keeps driving on the small model.

If the server refuses the comma after an update, check out the commit the fork
pins and rebuild:

```bash
git -C jetlink fetch
git -C jetlink checkout <commit from the fork's jetlink_repo entry>
```

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. On the server, `git pull`. On a Mac, restart `scripts/run-mac.sh`; delete
   `.venv` first if dependencies changed. On a Jetson, rebuild the image and
   point the service at it:

```bash
sudo docker/build.sh
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo systemctl restart jetlink-server
```

3. Plug in while parked and wait for the green icon. A new Jetlink or model may
   need another engine build. Cached engines stay valid across updates that do
   not change the model or runtime.

## Cutting a release

A `v*` tag is the whole release. The Release workflow builds the macOS app, the
Python sdist and wheel, and the Jetson image, and publishes them.

1. Bump `version` in `pyproject.toml` and commit. The tag has to match it: the
   workflow's first job is `macos/scripts/check-version.sh`, which fails in a
   second when they disagree.
2. Tag and push:

```bash
git tag v0.3.0
git push origin v0.3.0
```

3. Watch **Actions > Release**. The macOS job takes about 20 minutes (the
   embedded runtime, the build, notarization); the Jetson image runs under QEMU
   and takes up to 30.
4. Check the release page. It should carry `Jetlink-0.3.0.dmg`,
   `Jetlink-0.3.0.zip`, `SHA256SUMS`, the sdist and the wheel, and notes ending
   with the GHCR image line.

A prerelease tag is published as a prerelease. Both spellings are
recognised: a hyphen (`v0.3.0-rc1`) and the PEP 440 suffixes
(`v0.3.0a1`, `v0.3.0b2`, `v0.3.0rc1`). Prefer PEP 440. `pyproject.toml`
has to hold the version verbatim, and setuptools normalizes a hyphenated
version anyway, so `v0.3.0-alpha.1` would ship a wheel named
`jetlink-0.3.0a1` and the tag would not match its own artifacts.

### Installing the app

Download the DMG from the release, open it, and drag Jetlink to Applications.
`docs/macos-app.md` covers the first run.

Verify the download against `SHA256SUMS`:

```bash
shasum -a 256 -c SHA256SUMS
```

### The container images

Every release pushes two images to `ghcr.io/zoompilot/jetlink`, each tagged with
the full version and with `major.minor`, and each carrying the platform it is
for:

| Tag | Platform | Base |
| --- | --- | --- |
| `0.3.0-jetson` | linux/arm64, JetPack | `l4t-jetpack:r36.4.0` |
| `0.3.0-cuda` | linux/amd64, NVIDIA GPU | `nvidia/cuda:12.9.1-base-ubuntu24.04` |

The suffix is not decoration. `arm64` on its own does not mean Jetson: the
Jetson image wants the L4T stack and the nvidia container runtime, and an
ordinary arm64 server that pulled an unsuffixed tag would get something it
cannot run. Nothing takes `:latest`, so a `docker pull` with no tag fails
rather than handing out whichever release was cut last.

Pull the Jetson one on the Jetson instead of building, and point the service at
that image:

```bash
sudo docker pull ghcr.io/zoompilot/jetlink:0.3.0-jetson
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' ghcr.io/zoompilot/jetlink:0.3.0-jetson \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo systemctl restart jetlink-server
```

Either image job may fail without blocking the rest of the release; the release
notes then say which, and building locally still works.

### Signing secrets

With none of these set, the release still happens: the app is signed ad hoc and
published as `Jetlink-X.Y.Z-unsigned.zip` with no DMG, so a fork can cut its own
build. The workflow log's "Report the signing mode" step says which mode it ran
in, and the release notes carry the Gatekeeper instructions for an unsigned one.

| Secret | What |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | the Developer ID Application certificate with its private key, exported from Keychain Access as a .p12 and base64 encoded |
| `MACOS_CERTIFICATE_PASSWORD` | the .p12 password |
| `KEYCHAIN_PASSWORD` | any random string; it locks the temporary keychain the runner builds in |
| `APPLE_TEAM_ID` | the 10 character team id, passed to the build as `DEVELOPMENT_TEAM` |
| `NOTARY_KEY_ID` | the App Store Connect API key id |
| `NOTARY_ISSUER_ID` | the issuer id of that key |
| `NOTARY_PRIVATE_KEY_P8_BASE64` | the key's .p8 file, base64 encoded |

The notary key is a Team key with the Developer role, made under **Users and
Access > Integrations** in App Store Connect. `base64 -i cert.p12 | pbcopy`
produces what the two base64 secrets want.

## Rolling back

Turn off **Settings > Models > Accelerator Link** to stop using Jetlink
immediately. To roll back properly, restore the previous comma build and the
previous server image together; restoring one side can leave them incompatible.
Keep the model cache. On the Jetson, `docker image ls` shows earlier images,
and the previous image ID can be written back into `/etc/jetlink/server.env`.
