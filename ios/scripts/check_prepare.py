#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Does the Swift model preparation produce the model the Python one does?

The iPhone prepares what the comma uploads with a Swift port of
jetlink/onnx_patch.py (ios/JetlinkKit/Sources/JetlinkKit/OnnxPrepare.swift).
This runs both on one ONNX, for each CoreML device (the Neural Engine's
reference being ane_passes.py), and requires the two
results to be the same protobuf message field for field: every node, every
initializer's bytes, every value_info, the opsets, the functions, the
metadata. Only the COREML_CACHE_KEY value is set to match, since each side
derives it from its own artifact path.

    swift build -c release --package-path ios/JetlinkKit
    python3 ios/scripts/check_prepare.py --onnx big_driving_supercombo.onnx \\
        --swift ios/JetlinkKit/.build/release/jetlink-swift

Needs the onnx package, and about four times the model's size in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ane_passes import ane_prepared_model
from jetlink.server.backends.ort import COREML_CACHE_KEY, _prepared_model, _with_cache_key

KEY = 'checkprepare'


def python_prepared(src: Path, device: str, dst: Path) -> None:
  # The Neural Engine build is the Mac's GPU preparation plus the phone's
  # passes in ane_passes.py, not the Mac's fp32 LayerNorms; see there.
  if device == 'ane':
    model = ane_prepared_model(src)
  else:
    model = _prepared_model(src, for_coreml=device == 'coreml')
  onnx.save(_with_cache_key(model, KEY), str(dst))


def digest(t) -> str:
  return hashlib.sha256(t.raw_data).hexdigest()[:16] if t.raw_data else 'typed'


def first_difference(a: onnx.ModelProto, b: onnx.ModelProto) -> str | None:
  """A readable account of the first place two models differ, or None."""
  ga, gb = a.graph, b.graph
  for what, xs, ys in (('node', ga.node, gb.node), ('input', ga.input, gb.input),
                       ('output', ga.output, gb.output), ('value_info', ga.value_info, gb.value_info)):
    if len(xs) != len(ys):
      return f"{len(xs)} vs {len(ys)} {what}s"
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
      if x != y:
        return f"{what} {i} differs:\n--- python\n{x}\n--- swift\n{y}"
  ia = [(t.name, list(t.dims), t.data_type, digest(t)) for t in ga.initializer]
  ib = [(t.name, list(t.dims), t.data_type, digest(t)) for t in gb.initializer]
  if ia != ib:
    if len(ia) != len(ib):
      return f"{len(ia)} vs {len(ib)} initializers"
    for x, y in zip(ia, ib, strict=True):
      if x != y:
        return f"initializer differs: python {x} swift {y}"
  for i, (x, y) in enumerate(zip(ga.initializer, gb.initializer, strict=True)):
    if x != y:
      return f"initializer {i} ({x.name}) differs in a field other than its data"
  for field in ('opset_import', 'metadata_props', 'functions'):
    if list(getattr(a, field)) != list(getattr(b, field)):
      return f"{field} differs:\n  python {list(getattr(a, field))}\n  swift  {list(getattr(b, field))}"
  if a != b:
    return "the models differ somewhere outside the graph's nodes, tensors and value infos"
  return None


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--onnx', required=True, type=Path)
  p.add_argument('--swift', required=True, type=Path, help='the jetlink-swift binary')
  p.add_argument('--device', choices=('coreml', 'ane', 'cpu'), action='append',
                 help='repeatable; default coreml, ane and cpu')
  args = p.parse_args()

  failed = 0
  for device in args.device or ['coreml', 'ane', 'cpu']:
    with tempfile.TemporaryDirectory() as tmp:
      py, sw = Path(tmp) / 'python.onnx', Path(tmp) / 'swift.onnx'
      t0 = time.time()
      python_prepared(args.onnx, device, py)
      t1 = time.time()
      subprocess.run([str(args.swift), 'prepare', str(args.onnx), str(sw), '--device', device,
                      '--cache-key', KEY], check=True)
      t2 = time.time()
      a, b = onnx.load(str(py)), onnx.load(str(sw))
      onnx.checker.check_model(b, full_check=False)
      diff = first_difference(a, b)
      print(f"{device}: python {t1 - t0:.1f} s, swift {t2 - t1:.1f} s, "
            f"{len(b.graph.node)} nodes, {len(b.graph.initializer)} initializers: "
            f"{'identical' if diff is None else 'DIFFERENT'}")
      if diff is not None:
        print(diff)
        failed += 1
      # the key is what the CoreML cache is found by
      if device != 'cpu' and [p.value for p in b.metadata_props if p.key == COREML_CACHE_KEY] != [KEY]:
        print(f"{device}: swift output lacks the cache key")
        failed += 1
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
