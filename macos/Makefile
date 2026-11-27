# Jetlink for Mac.
#
#   make app      the whole thing: project, build, sign
#   make smoke    launch the built app and talk to its server over TCP
#   make appcast  sign the DMG for Sparkle and write its update feed
#   make dev      open the project in Xcode
#
# Every recipe runs under bash with `set -euo pipefail`, so a failing command
# in the middle of a recipe fails the target.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

MACOS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT := $(patsubst %/,%,$(dir $(MACOS_DIR)))
BUILD := $(MACOS_DIR)/build
APP := $(BUILD)/Jetlink.app

# A tag like v0.2.0 becomes 0.2.0. An untagged tree makes `describe --always`
# return a bare commit sha, which is not a legal CFBundleShortVersionString, so
# anything that does not look like a dotted version becomes 0.0.0. Only v*
# tags count: the release workflow moves `edge` to main's tip, and on a release
# commit describe would pick it over the version tag.
# No --dirty: the build regenerates the project, and an xcodegen other than
# the one that wrote the committed copy leaves the tree changed, which named
# v0.10.1 and v0.10.2 "-dirty" in the DMG and the update feed.
JETLINK_VERSION ?= $(or $(shell git -C $(MACOS_DIR) describe --tags --match 'v[0-9]*' --always 2>/dev/null | sed -E -e 's/^v//' -e '/^[0-9]+\./!s/.*/0.0.0/'),0.0.0)
JETLINK_BUILD ?= $(shell git -C $(MACOS_DIR) rev-list --count HEAD 2>/dev/null || echo 1)
SIGN_IDENTITY ?= -
export JETLINK_VERSION
export JETLINK_BUILD

.PHONY: all project app sign notarize dmg appcast test smoke dev clean

all: app

# Generated with the placeholder version, never the one from git: the project
# is committed, and baking `git rev-list --count HEAD` into it would make it
# stale again on every commit. The real version reaches the build as an
# xcodebuild command line setting (scripts/build-app.sh).
project:
	JETLINK_VERSION=0.0.0 JETLINK_BUILD=1 xcodegen generate --spec $(MACOS_DIR)/project.yml --project $(MACOS_DIR)

app: project
	SIGN_IDENTITY="$(SIGN_IDENTITY)" $(MACOS_DIR)/scripts/build-app.sh
	SIGN_IDENTITY="$(SIGN_IDENTITY)" $(MACOS_DIR)/scripts/sign.sh $(APP)

sign:
	SIGN_IDENTITY="$(SIGN_IDENTITY)" $(MACOS_DIR)/scripts/sign.sh $(APP)

notarize:
	$(MACOS_DIR)/scripts/notarize.sh $(APP)

dmg:
	SIGN_IDENTITY="$(SIGN_IDENTITY)" $(MACOS_DIR)/scripts/make-dmg.sh $(APP)

# After dmg. Needs SPARKLE_ED_PRIVATE_KEY or SPARKLE_ED_KEY_FILE.
appcast:
	$(MACOS_DIR)/scripts/make-appcast.sh $(APP)

test: project
	xcodebuild test -project $(MACOS_DIR)/Jetlink.xcodeproj -scheme Jetlink \
	  -destination 'platform=macOS' -derivedDataPath $(BUILD)/DerivedData \
	  CODE_SIGN_IDENTITY="$(SIGN_IDENTITY)" CODE_SIGNING_ALLOWED=YES

# The built bundle's server, over TCP, with a cache of its own: no comma and no
# model. The client is a few lines of jetlink's protocol; any python3 with no
# packages installed runs it.
smoke:
	python3 $(MACOS_DIR)/scripts/smoke.py --app $(APP)

dev: project
	open $(MACOS_DIR)/Jetlink.xcodeproj

# build/python and build/downloads are what the app's embedded Python runtime,
# gone since this release, left behind; clean takes them too.
clean:
	rm -rf $(BUILD)/Jetlink.app $(BUILD)/DerivedData $(BUILD)/python $(BUILD)/downloads $(BUILD)/dmg-stage
	rm -f $(BUILD)/Jetlink-*.dmg $(BUILD)/Jetlink-*.zip $(BUILD)/SHA256SUMS $(BUILD)/icon.png $(BUILD)/appcast.xml
