# Jetlink for iPhone.
#
#   make project   regenerate Jetlink.xcodeproj from project.yml (needs xcodegen)
#   make test      JetlinkKit's tests, on this Mac
#   make check     the Swift preparation against the Python one (needs the repo's .venv)
#   make device    build the app for a phone, unsigned, to see it compiles
#   make server    the phone's server as a Mac command: ios/JetlinkKit/.build/release/jetlink-swift
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

# Run from this directory, or as `make -C ios`. CURDIR, not make's path
# functions, because those split a path with a space in it.
IOS_DIR := $(CURDIR)
PYTHON ?= $(IOS_DIR)/../.venv/bin/python
SWIFT_BIN := $(IOS_DIR)/JetlinkKit/.build/release/jetlink-swift

.PHONY: project test check device server

project:
	cd "$(IOS_DIR)" && xcodegen generate --spec project.yml --project .

test:
	swift test --package-path "$(IOS_DIR)/JetlinkKit"

server:
	swift build -c release --package-path "$(IOS_DIR)/JetlinkKit"

check: server
	"$(PYTHON)" "$(IOS_DIR)/scripts/prepare_variants.py" --swift "$(SWIFT_BIN)"

device:
	xcodebuild -project "$(IOS_DIR)/Jetlink.xcodeproj" -scheme Jetlink -destination 'generic/platform=iOS' \
	  -configuration Release -derivedDataPath "$(IOS_DIR)/build/DerivedData" CODE_SIGNING_ALLOWED=NO build
