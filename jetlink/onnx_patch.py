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

# tinygrad's exporter can leave a layout hint in the graph as a node in its own
# domain. Contiguous means "materialise this tensor", which is a statement about
# tinygrad's internal buffers and nothing about the arithmetic, so it is safe to
# bypass - and it has to be, because TensorRT's parser rejects any op outside a
# domain it knows. comma's 2026-09-01 model carries exactly one; the model a day
# earlier carries none, so this comes and goes with how a model was exported.
TINYGRAD_DOMAIN = 'org.tinygrad'
PASSTHROUGH_OPS = ('Contiguous',)


def needs_patch(model: onnx.ModelProto) -> bool:
  return any(vi.name in IMG_INPUTS and vi.type.tensor_type.elem_type == TensorProto.UINT8
             for vi in model.graph.input)


def strip_tinygrad_ops(model: onnx.ModelProto) -> int:
  """Bypass tinygrad's layout-hint nodes. In place, returns how many went.

  Refuses anything it has not been told is a passthrough rather than guessing:
  silently dropping an op that did something would change what the car sees.
  """
  g = model.graph
  graph_outputs = {o.name for o in g.output}
  removed = 0

  for node in [n for n in g.node if n.domain == TINYGRAD_DOMAIN]:
    if node.op_type not in PASSTHROUGH_OPS:
      raise ValueError(f"unknown {TINYGRAD_DOMAIN} op {node.op_type!r}; it may not "
                       "be a no-op, so dropping it is not safe")
    if len(node.input) != 1 or len(node.output) != 1 or node.attribute:
      raise ValueError(f"{node.op_type} is not a plain one-in one-out passthrough")

    source, produced = node.input[0], node.output[0]
    for n in g.node:
      for i, name in enumerate(n.input):
        if name == produced:
          n.input[i] = source
    for o in g.output:
      if o.name == produced:
        # It fed a graph output directly, so the producer has to take that name.
        o.name = source
    if produced in graph_outputs:
      for n in g.node:
        for i, name in enumerate(n.output):
          if name == source:
            n.output[i] = produced
    g.node.remove(node)
    removed += 1

  if removed:
    for i, opset in enumerate(model.opset_import):
      if opset.domain == TINYGRAD_DOMAIN:
        del model.opset_import[i]
        break
    live = {n for node in g.node for n in list(node.input) + list(node.output)}
    for i in reversed(range(len(g.value_info))):
      if g.value_info[i].name not in live:
        del g.value_info[i]
  return removed


def patch_uint8_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
  """Retype the uint8 image inputs to fp16 and drop the head Cast. In place."""
  g = model.graph

  img_inputs = [vi for vi in g.input if vi.name in IMG_INPUTS]
  if not img_inputs:
    raise ValueError(f"model has none of {IMG_INPUTS} as graph inputs")

  # Two graph shapes in the wild, and exporters pick between them freely:
  #
  #   comma      Concat(img, big_img) -> cat -> Cast(fp16)
  #   sunnypilot Cast(img), Cast(big_img) -> Concat -> cat
  #
  # Either way the fix is the same, because the casts are the only thing
  # standing between a uint8 graph input and the fp16 the rest of the model
  # wants: declare the inputs fp16 and delete the casts.
  casts = _head_casts(g)
  if not casts:
    raise ValueError("could not find the head Cast on the image inputs; "
                     + "neither a Cast per input nor one after their Concat")

  for cast in casts:
    to = next(onnx.helper.get_attribute_value(a) for a in cast.attribute if a.name == 'to')
    if to != TensorProto.FLOAT16:
      raise ValueError(f"head Cast targets {to}, expected FLOAT16 ({TensorProto.FLOAT16})")

  for vi in img_inputs:
    if vi.type.tensor_type.elem_type != TensorProto.UINT8:
      raise ValueError(f"{vi.name} is not uint8; model already patched?")
    vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  retyped = set()
  for cast in casts:
    # Everything downstream reads the cast's source directly now.
    source, produced = cast.input[0], cast.output[0]
    for n in g.node:
      for i, name in enumerate(n.input):
        if name == produced:
          n.input[i] = source
    g.node.remove(cast)
    retyped.add(source)

  # A declared type left saying uint8 is not merely untidy: TensorRT tolerates
  # it but onnxruntime rejects the model outright, and we want both to load.
  for vi in g.value_info:
    if vi.name in retyped:
      vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  return model


def _head_casts(g) -> list:
  """The Cast nodes turning the uint8 image inputs into fp16, in either shape."""
  per_input = [n for n in g.node
               if n.op_type == 'Cast' and len(n.input) == 1 and n.input[0] in IMG_INPUTS]
  if per_input:
    return per_input

  concat = next((n for n in g.node if n.op_type == 'Concat'
                 and all(i in IMG_INPUTS for i in n.input)), None)
  if concat is None:
    return []
  cat = concat.output[0]
  cast = next((n for n in g.node if n.op_type == 'Cast' and list(n.input) == [cat]), None)
  return [cast] if cast is not None else []


def patch_file(src: str, dst: str, check: bool = True) -> str:
  model = onnx.load(src)
  strip_tinygrad_ops(model)
  if needs_patch(model):
    patch_uint8_inputs(model)
  if check:
    onnx.checker.check_model(model, full_check=False)
  onnx.save(model, dst)
  return dst


if __name__ == '__main__':
  import sys
  print(patch_file(sys.argv[1], sys.argv[2]))
