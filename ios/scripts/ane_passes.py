"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The iPhone's Neural Engine preparation, in Python, as the reference the
Swift one (OnnxPrepare.swift) is checked against by check_prepare.py.

The Mac server's Neural Engine build runs everything after the vision trunk
on the GPU. That costs little on a Mac's GPU and a great deal on a phone's,
so these passes keep the whole model on the Neural Engine, in one piece:

  expand_to_tile        (in _prepared_model, for every CoreML build)
                        Expand(x, [1, 1, 32, 1]) on a [1, 9, 1, 512] x is
                        Tile(x, [1, 1, 32, 1]): the same values, an op the
                        CoreML provider accepts, so the graph stays one
                        partition.
  prescale_layernorm    LayerNorm(x / 8) is LayerNorm(x) up to epsilon, and
                        keeps the fp16 arithmetic in range: the policy's
                        inputs reach 1189, whose square overflows fp16 while
                        the row sums of (x / 8)^2 stay under 34,000. Epsilon
                        is left as it is. Measured over 32 recurrent frames
                        of the 766 MB model, 2026-09-24.
  heads_in_fp32         the small MLPs between the end of the vision trunk
                        and the output (the summarizer and hydra heads that
                        make road_transform, pose, lane_lines_prob and the
                        rest) in fp32, which puts them on the GPU or CPU.
                        In fp16 on an iPhone 17 Pro's Neural Engine,
                        road_transform failed the parity gate. 2026-09-25.

On the M1 Pro: 99.5% of the estimated cost on the Neural Engine (was the
policy on the GPU), the parity gate passes, 30 ms back to back.

The preparation is _prepared_model(for_coreml=True), which includes
expand_to_tile, with the other two after it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from jetlink.server.backends.ort import _prepared_model

# The phone serves queued graphs (V1, V2), whose images are these two.
QUEUED_IMG_INPUTS = ('img', 'big_img')


# vision_nodes and _static_info as jetlink.onnx_patch had them, which the
# Swift preparation (OnnxPrepare.visionNodes, OnnxModel.staticInfo) mirrors.
def vision_nodes(model: onnx.ModelProto) -> set[str]:
  """Names of the nodes that depend on the image inputs alone: the vision
  trunk and the heads that hang off it. Found by dataflow: a node is vision
  when every tensor it reads is an image input, an initializer, or another
  vision node's output. Nodes reading only initializers count as neither."""
  g = model.graph
  init = {t.name for t in g.initializer}
  image_inputs = {vi.name for vi in g.input if vi.name in QUEUED_IMG_INPUTS}
  if not image_inputs:
    raise ValueError(f"model has none of {QUEUED_IMG_INPUTS} as graph inputs")
  vision_tensors = set(image_inputs)
  names: set[str] = set()
  for node in g.node:
    data = [x for x in node.input if x and x not in init]
    if data and all(x in vision_tensors for x in data):
      names.add(node.name)
      vision_tensors.update(node.output)
  return names


def _static_info(model: onnx.ModelProto, wanted: set[str]) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
  """Static shapes and element types of the graph's tensors, from what the
  file carries; the shape inferrer only when one of `wanted` is missing.
  Fails open: a tensor still unknown afterwards is simply absent, and the
  caller decides what it cannot do without it."""
  g = model.graph
  dims: dict[str, tuple[int, ...]] = {}
  dtypes: dict[str, int] = {}

  def take(values):
    for vi in values:
      tt = vi.type.tensor_type
      if tt.elem_type:
        dtypes[vi.name] = tt.elem_type
      if tt.HasField('shape'):
        dims[vi.name] = tuple(d.dim_value if d.HasField('dim_value') else -1 for d in tt.shape.dim)
  take(g.input)
  take(g.value_info)
  take(g.output)
  if wanted - {n for n in dims if n in dtypes}:
    try:
      take(onnx.shape_inference.infer_shapes(model).graph.value_info)
    except Exception:
      pass
  return dims, dtypes


LAYERNORM_PRESCALE = 8
# The ops heads_in_fp32 moves into fp32: the small MLPs and linear layers that
# end the vision trunk. A reduction, a reshape or anything else ends the region.
HEAD_OPS = {'Gemm', 'MatMul', 'LayerNormalization', 'Gelu', 'Add', 'Sub', 'Mul', 'Div', 'Relu', 'Sigmoid', 'Tanh'}
HEAD_MAX_NODES = 64


def prescale_layernorm(model, only: set[str], k: int = LAYERNORM_PRESCALE) -> int:
  """Feed the fp16 LayerNorms named in `only` their input times 1/k, one Mul
  per distinct input. In place; returns how many."""
  g = model.graph
  const = f"__layernorm_prescale_{k}"
  _, dtypes = _static_info(model, {n.input[0] for n in g.node if n.op_type == 'LayerNormalization'})
  new, scaled, done = [], {}, 0
  for n in g.node:
    if n.op_type == 'LayerNormalization' and n.name in only and dtypes.get(n.input[0]) == TensorProto.FLOAT16:
      x = n.input[0]
      if x not in scaled:
        scaled[x] = f"{x}__scaled"
        new.append(helper.make_node('Mul', [x, const], [scaled[x]], name=f"{n.name}__prescale"))
      n.input[0] = scaled[x]
      done += 1
    new.append(n)
  if done:
    g.initializer.append(numpy_helper.from_array(np.array(1.0 / k, np.float16), const))
    del g.node[:]
    g.node.extend(new)
  return done


def vision_heads(model, vision: set[str]) -> list[int]:
  """The indices, in graph order, of the heads that end the vision trunk:
  the largest set of vision nodes with ops in HEAD_OPS whose outputs are all
  read, and read only, by each other or by a Concat that makes a graph
  output. Empty when there is no such Concat or the set would exceed
  HEAD_MAX_NODES."""
  g = model.graph
  outputs = {o.name for o in g.output}
  ends = {i for i, n in enumerate(g.node) if n.op_type == 'Concat' and any(o in outputs for o in n.output)}
  if not ends:
    return []
  readers: dict[str, set[int]] = {}
  for i, n in enumerate(g.node):
    for x in n.input:
      readers.setdefault(x, set()).add(i)
  region: set[int] = set()
  grew = True
  while grew:
    grew = False
    for i in reversed(range(len(g.node))):
      n = g.node[i]
      if i in region or n.name not in vision or n.op_type not in HEAD_OPS or any(o in outputs for o in n.output):
        continue
      read = [readers.get(o, set()) for o in n.output]
      if all(r and r <= region | ends for r in read):
        region.add(i)
        grew = True
  return sorted(region) if len(region) <= HEAD_MAX_NODES else []


def heads_in_fp32(model, vision: set[str]) -> int:
  """Run vision_heads in fp32: cast what they read from the trunk up, their
  fp16 weights to fp32 copies, and what they hand the output Concat back
  down. The Neural Engine cannot run fp32, so CoreML places them on the GPU
  or CPU. In place; returns how many nodes moved.

  The heads are small (24 nodes and 4 MB of weights in the 766 MB chestnut
  model), but in fp16 on the Neural Engine their LayerNorm, Gelu and 1024-wide
  Gemms lose enough that road_transform fails the parity gate on an iPhone
  17 Pro (worst column 0.9989). Computed exactly from the Neural Engine's own
  trunk output, every column is 0.9996 or better. Measured 2026-09-25."""
  g = model.graph
  index = vision_heads(model, vision)
  if not index:
    return 0
  heads = set(index)
  init = {t.name: t for t in g.initializer}
  produced = {o for i in index for o in g.node[i].output}
  entries = {x for i in index for x in g.node[i].input if x and x not in init and x not in produced}
  _, dtypes = _static_info(model, entries)
  if any(dtypes.get(x) != TensorProto.FLOAT16 for x in entries):
    return 0
  weights = {x for i in index for x in g.node[i].input if x in init}
  if any(init[w].data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT) for w in weights):
    return 0
  read_outside = {x for j, n in enumerate(g.node) if j not in heads for x in n.input}
  exits = produced & read_outside
  wide = {}
  for w in sorted(weights):
    if init[w].data_type == TensorProto.FLOAT16:
      wide[w] = f"{w}__fp32"
      # layernorm_in_fp32 names its copies the same way; one is enough
      if wide[w] not in init:
        g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(init[w]).astype(np.float32), wide[w]))
  new, cast = [], set()
  for j, n in enumerate(g.node):
    if j not in heads:
      new.append(n)
      continue
    for i, x in enumerate(n.input):
      if x in entries:
        if x not in cast:
          cast.add(x)
          new.append(helper.make_node('Cast', [x], [f"{x}__fp32"], name=f"{x}__cast_fp32", to=TensorProto.FLOAT))
        n.input[i] = f"{x}__fp32"
      elif x in exits:
        n.input[i] = f"{x}__fp32"
      elif x in wide:
        n.input[i] = wide[x]
    back = [o for o in n.output if o in exits]
    for i, o in enumerate(n.output):
      if o in exits:
        n.output[i] = f"{o}__fp32"
    new.append(n)
    for o in back:
      new.append(helper.make_node('Cast', [f"{o}__fp32"], [o], name=f"{o}__cast_fp16", to=TensorProto.FLOAT16))
  del g.node[:]
  g.node.extend(new)
  # The fp16 originals, unless something outside the heads reads them too.
  keep = [t for t in g.initializer if t.name not in wide or t.name in read_outside]
  del g.initializer[:]
  g.initializer.extend(keep)
  # What the heads compute inside is fp32 now; the value infos said fp16.
  keep_vi = [v for v in g.value_info if v.name not in produced - exits]
  del g.value_info[:]
  g.value_info.extend(keep_vi)
  return len(index)


def ane_prepared_model(onnx_path: Path):
  """The model the iPhone's Neural Engine build hands onnxruntime."""
  model = _prepared_model(Path(onnx_path), for_coreml=True)   # includes expand_to_tile
  vision = vision_nodes(model)
  policy = {n.name for n in model.graph.node} - vision
  prescale_layernorm(model, policy)
  heads_in_fp32(model, vision)
  return model
