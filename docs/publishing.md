# Publish a release

For updating an installed server, see [updates and rollback](releasing.md).
This page is for maintainers publishing artifacts.

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

## Installing the app

For a ZIP, unzip it and drag Jetlink.app to Applications. For a signed DMG,
open it and drag Jetlink to Applications. Unsigned builds provide a ZIP only.
See the [Mac guide](macos-app.md) for first launch instructions.

Verify the download against `SHA256SUMS`:

```bash
shasum -a 256 -c SHA256SUMS
```

## The container images

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

## Signing secrets

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
