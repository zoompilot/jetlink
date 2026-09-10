# Jetlink for Mac.
#
#   make app      the whole thing: project, embedded runtime, build, sign
#   make python   just the embedded runtime (about 200 MB of downloads, cached)
#   make dev      open the project in Xcode against the repo's .venv
#
# Every recipe runs under bash with `set -euo pipefail`, so a failing command
# in the middle of a recipe fails the target.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

MACOS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT := $(patsubst %/,%,$(dir $(MACOS_DIR)))
BUILD := $(MACOS_DIR)/build
APP := $(BUILD)/Jetlink.app
PYTHON_PREFIX := $(BUILD)/python

# A tag like v0.2.0 becomes 0.2.0. An untagged tree makes `describe --always`
# return a bare commit sha, which is not a legal CFBundleShortVersionString, so
# anything that does not look like a dotted version becomes 0.0.0.
JETLINK_VERSION ?= $(or $(shell git -C $(MACOS_DIR) describe --tags --always --dirty 2>/dev/null | sed -E -e 's/^v//' -e '/^[0-9]+\./!s/.*/0.0.0/'),0.0.0)
JETLINK_BUILD ?= $(shell git -C $(MACOS_DIR) rev-list --count HEAD 2>/dev/null || echo 1)
SIGN_IDENTITY ?= -
export JETLINK_VERSION
export JETLINK_BUILD

# The interpreter `make dev` hands to Xcode: the repo's development venv, which
# needs `pip install -e ".[ort,usb]"` plus tinygrad from git.
JETLINK_PYTHON ?= $(REPO_ROOT)/.venv/bin/python

.PHONY: all project python app sign notarize dmg test smoke dev clean

all: app

# Generated with the placeholder version, never the one from git: the project
# is committed, and baking `git rev-list --count HEAD` into it would make it
# stale again on every commit. The real version reaches the build as an
# xcodebuild command line setting (scripts/build-app.sh).
project:
	JETLINK_VERSION=0.0.0 JETLINK_BUILD=1 xcodegen generate --spec $(MACOS_DIR)/project.yml --project $(MACOS_DIR)

python:
	$(MACOS_DIR)/Python/embed-python.sh

app: project python
	SIGN_IDENTITY=$(SIGN_IDENTITY) $(MACOS_DIR)/scripts/build-app.sh
	SIGN_IDENTITY=$(SIGN_IDENTITY) $(MACOS_DIR)/scripts/sign.sh $(APP)

sign:
	SIGN_IDENTITY=$(SIGN_IDENTITY) $(MACOS_DIR)/scripts/sign.sh $(APP)

notarize:
	$(MACOS_DIR)/scripts/notarize.sh $(APP)

dmg:
	SIGN_IDENTITY=$(SIGN_IDENTITY) $(MACOS_DIR)/scripts/make-dmg.sh $(APP)

test: project
	xcodebuild test -project $(MACOS_DIR)/Jetlink.xcodeproj -scheme Jetlink \
	  -destination 'platform=macOS' -derivedDataPath $(BUILD)/DerivedData \
	  CODE_SIGN_IDENTITY="$(SIGN_IDENTITY)" CODE_SIGNING_ALLOWED=YES

# The embedded runtime as the app will run it: an environment built from
# scratch, never inherited (01-contracts.md section 9).
smoke:
	test -x $(APP)/Contents/Resources/python/bin/python3 || { echo "no embedded runtime in $(APP); run make app" >&2; exit 1; }
	env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$$HOME" TMPDIR="$${TMPDIR:-/tmp}" \
	  LANG=en_US.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
	  $(APP)/Contents/Resources/python/bin/python3 -m jetlink.server.main --list-backends
	env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$$HOME" TMPDIR="$${TMPDIR:-/tmp}" \
	  LANG=en_US.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
	  $(APP)/Contents/Resources/python/bin/python3 -c "import usb1; usb1.USBContext(); print('usb1 ok')"

dev: project
	JETLINK_PYTHON=$(JETLINK_PYTHON) open $(MACOS_DIR)/Jetlink.xcodeproj

# Keeps build/downloads: the interpreter tarball and the wheels are 200 MB and
# never change for a given pin.
clean:
	rm -rf $(BUILD)/Jetlink.app $(BUILD)/DerivedData $(BUILD)/python $(BUILD)/dmg-stage
	rm -f $(BUILD)/Jetlink-*.dmg $(BUILD)/Jetlink-*.zip $(BUILD)/SHA256SUMS $(BUILD)/icon.png
