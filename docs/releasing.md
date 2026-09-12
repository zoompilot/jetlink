# Updates and rollback

The comma build and the Jetlink server must be compatible. Update both while
parked.

## Which Jetlink to run

Clone `main`. The zoompilot fork records the exact Jetlink commit it was tested
with as its `jetlink_repo` submodule, and `main` is kept compatible with the
current `jetson-trt` branch. If the protocol versions differ, the server rejects
the connection and the comma keeps driving on the small model.

If the server refuses the comma after an update, check out the commit recorded
in the fork's `jetlink_repo` submodule and rebuild. From the parent directory of
your Jetlink checkout, run the commands below. Replace `COMMIT` with that hash:

```bash
git -C jetlink fetch
git -C jetlink checkout COMMIT
```

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. Update the server using the method you installed:

   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - Source install: run `git pull` from the Jetlink checkout. For the Mac
     script, restart `scripts/run-mac.sh`; recreate `.venv` if dependencies changed.
   - Jetson source install: run `git pull`, rebuild the image, and update the service
     with the commands below from the checkout:

```bash
sudo docker/build.sh
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' jetlink:latest \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo systemctl restart jetlink-server
```

3. Plug in while parked and wait for the green icon. A new Jetlink or model may
   need another engine build. Cached engines stay valid across updates that do
   not change the model or runtime.

## Publish a release (maintainers)

Pushing a `v*` tag starts the Release workflow. It builds and publishes the
macOS app, Python source distribution and wheel, and container images.

1. Update `version` in `pyproject.toml` and commit. The tag version must match it.
   `macos/scripts/check-version.sh` checks this before the build.
2. Tag and push. Replace `0.3.0` with the version you set:

```bash
git tag v0.3.0
git push origin v0.3.0
```

3. Watch **Actions > Release**. The macOS job takes about 20 minutes (the
   embedded runtime, the build, notarization); the Jetson image runs under QEMU
   and takes up to 30.
4. Check the release page. It should include `Jetlink-0.3.0.dmg`,
   `Jetlink-0.3.0.zip`, `SHA256SUMS`, the sdist and the wheel, and notes ending
   with the GHCR image line.

A prerelease tag is published as a prerelease. Supported formats include a
hyphen (`v0.3.0-rc1`) and the PEP 440 suffixes (`v0.3.0a1`, `v0.3.0b2`,
`v0.3.0rc1`). Use the same version string in `pyproject.toml` and the tag,
excluding the leading `v`. Prefer PEP 440 suffixes to avoid wheel filename
normalization.

### Installing the app

Download the DMG from the release, open it, and drag Jetlink to Applications.
See the [Mac guide](macos-app.md) for first launch instructions.

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

Choose the suffix for your platform. The Jetson image requires L4T and the
NVIDIA container runtime; it does not support generic ARM64 servers. Always
specify a version and suffix. The registry does not publish a `latest` tag.

To update a Jetson using a release image, run the commands below on the Jetson.
Replace `0.3.0` with the release version:

```bash
sudo docker pull ghcr.io/zoompilot/jetlink:0.3.0-jetson
sudo docker image inspect --format 'JETLINK_IMAGE={{.Id}}' ghcr.io/zoompilot/jetlink:0.3.0-jetson \
  | sudo tee /etc/jetlink/server.env >/dev/null
sudo systemctl restart jetlink-server
```

Either image job may fail without blocking the rest of the release; the release
notes then say which, and building locally still works.

### Signing secrets

Without these secrets, the app is signed ad hoc and published as
`Jetlink-X.Y.Z-unsigned.zip` with no DMG, so forks can publish unsigned builds.
The workflow log's "Report the signing mode" step says which mode it ran in, and
the release notes carry the Gatekeeper instructions for an unsigned one.

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
produces a base64-encoded value for a certificate secret. Use the `.p8` file
instead for `NOTARY_PRIVATE_KEY_P8_BASE64`.

## Rolling back

Turn off **Settings > Models > Accelerator Link** to stop using Jetlink
immediately. To roll back, restore the previous comma build and the previous
server image together; restoring one side can leave them incompatible. Keep the
model cache. On the Jetson, `docker image ls` shows earlier images, and the
previous image ID can be written back into `/etc/jetlink/server.env`.
