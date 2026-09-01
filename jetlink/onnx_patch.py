"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Make an openpilot driving model acceptable to TensorRT's ONNX parser.

TensorRT 10.3 rejects UINT8 graph inputs outright:

    Assertion failed: false: Found unsupported input type of UINT8

openpilot's graph head is `img, big_img (uint8) -> Concat -> Cast(to=fp16)`,
so the fix is to declare both image inputs FP16 and delete the now-redundant
Cast. The caller then feeds 0..255 as fp16, which is exact (every integer up to
2048 is representable) and costs nothing: the weights are already FP16.

Runs on the Jetson at engine-build time only. The model openpilot ships is
never modified in place.
"""
from __future__ import annotations

import onnx
from onnx import TensorProto

IMG_INPUTS = ('img', 'big_img')


def needs_patch(model: onnx.ModelProto) -> bool:
  return any(vi.name in IMG_INPUTS and vi.type.tensor_type.elem_type == TensorProto.UINT8
             for vi in model.graph.input)


def patch_uint8_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
  """Retype the uint8 image inputs to fp16 and drop the head Cast. In place."""
  g = model.graph

  img_inputs = [vi for vi in g.input if vi.name in IMG_INPUTS]
  if not img_inputs:
    raise ValueError(f"model has none of {IMG_INPUTS} as graph inputs")

  concat = next((n for n in g.node if n.op_type == 'Concat'
                 and all(i in IMG_INPUTS for i in n.input)), None)
  if concat is None:
    raise ValueError("could not find the Concat joining the image inputs")
  cat = concat.output[0]

  cast = next((n for n in g.node if n.op_type == 'Cast' and list(n.input) == [cat]), None)
  if cast is None:
    raise ValueError(f"no Cast consuming the Concat output {cat!r}")
  to = next(onnx.helper.get_attribute_value(a) for a in cast.attribute if a.name == 'to')
  if to != TensorProto.FLOAT16:
    raise ValueError(f"head Cast targets {to}, expected FLOAT16 ({TensorProto.FLOAT16})")

  for vi in img_inputs:
    if vi.type.tensor_type.elem_type != TensorProto.UINT8:
      raise ValueError(f"{vi.name} is not uint8; model already patched?")
    vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  # Rewire everything downstream of the Cast onto the Concat output, then drop it.
  for n in g.node:
    for i, name in enumerate(n.input):
      if name == cast.output[0]:
        n.input[i] = cat
  g.node.remove(cast)

  # The Concat output's declared type is stale (uint8) now. TensorRT tolerates
  # it; onnxruntime rejects the model outright, and we want both to load.
  for vi in g.value_info:
    if vi.name == cat:
      vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  return model


def patch_file(src: str, dst: str, check: bool = True) -> str:
  model = onnx.load(src)
  if needs_patch(model):
    patch_uint8_inputs(model)
  if check:
    onnx.checker.check_model(model, full_check=False)
  onnx.save(model, dst)
  return dst


if __name__ == '__main__':
  import sys
  print(patch_file(sys.argv[1], sys.argv[2]))
