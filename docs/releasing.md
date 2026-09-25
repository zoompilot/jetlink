# Updates and rollback

The comma build and the Jetlink server must be compatible. Update both while
parked.

## Which Jetlink to run

`main`. The installer installs it by default, and the Mac app's releases are cut
from it. The zoompilot fork records the exact Jetlink commit it was tested with
as its `jetlink_repo` submodule, and `main` is kept compatible with the current
`jetson-trt` branch. If the protocol versions differ, the server rejects the
connection and the comma keeps driving on the small model.

To install a release, or the commit the fork records, instead of `main`, pass
it to the installer. Replace `v0.4.0` with a release tag:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

## Updating

1. Update the comma first from **Settings > Software** and let it reboot.
2. Update the server using the method you installed:

   - Jetson or Linux PC with the installer: `jetlink update`. It keeps your
     answers, fetches the newest `main` (or your `--ref`), and restarts the
     server.
   - Mac app: quit Jetlink, replace it with the new release, and reopen it.
   - Source install: run `git pull` from the Jetlink checkout. For the Mac
     script, restart `scripts/run-mac.sh`; recreate `.venv` if dependencies
     changed.

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
   embedded runtime, the build, notarization); each image builds on a native
   runner for its architecture.
4. Check the release page. It should include `Jetlink-0.3.0.dmg`,
   `Jetlink-0.3.0.zip`, `SHA256SUMS`, the sdist and the wheel, and notes ending
   with the installer command and the GHCR image lines.

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
| `0.4.0-cuda` | linux/amd64 (NVIDIA PCs, driver 580+) and linux/arm64 (JetPack 7.2+) | `nvidia/cuda:13.2.1-base-ubuntu24.04` |
| `0.4.0-jetpack6`, also `0.4.0-jetson` | linux/arm64, JetPack 6 | `l4t-jetpack:r36.4.0` |

`-cuda` is one tag for two architectures, and Docker pulls the one for the
machine. The JetPack 6 image requires L4T r36 and the NVIDIA container runtime;
it does not run on JetPack 7 or generic Arm servers. The registry does not
publish a `latest` tag.

Each push to `main` also publishes `edge-cuda` and `edge-jetpack6` (the Docker
Images workflow), which is what the installer pulls. When neither exists for a
machine, the installer builds the image there instead.

Either image job may fail without blocking the rest of the release; the release
notes then say which, and `install.sh --build` still works.

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
server together; restoring one side can leave them incompatible. Keep the model
cache.

With the installer, run it with the release or commit to go back to:

```bash
curl -fsSL https://raw.githubusercontent.com/zoompilot/jetlink/v0.4.0/install.sh | bash -s -- --ref v0.4.0
```

Or put an earlier image back by hand: `sudo docker image ls` shows the images on
the machine, and the one to run is `JETLINK_IMAGE` in `/etc/jetlink/server.env`
(an image ID from `sudo docker image inspect --format '{{.Id}}' IMAGE`). Then
`jetlink restart`.
