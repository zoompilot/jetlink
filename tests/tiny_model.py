"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A driving-model-shaped ONNX small enough to build in a test.

The same inputs as the big model with the same names and dtypes (uint8 images
behind a Cast, fp16 everything else), an `org.tinygrad` Contiguous node in the
middle so the passthrough stripping and tinygrad's native handling are both
exercised, and openpilot's output_slices metadata. The arithmetic is a mean
over the images and one matmul, so a numpy reference is exact enough to judge
fp16 backends against.
"""
from __future__ import annotations

import codecs
import pickle
from pathlib import Path

import numpy as np

SHAPES = {
  'img': (1, 12, 8, 16),
  'big_img': (1, 12, 8, 16),
  'desire_pulse': (1, 33, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 32, 4, 8),
}
N_OUT = 64
SLICES = {'plan': slice(0, 16), 'lead_prob': slice(16, 19), 'hidden_state': slice(32, 64)}
FEATURES = 24 + 33 * 8 + 2 + 2 + 32 * 4 * 8   # the concat the matmul reads


def weights(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  w = (rng.standard_normal((FEATURES, N_OUT)) * 0.05).astype(np.float16)
  b = (rng.standard_normal(N_OUT) * 0.1).astype(np.float16)
  return w, b


def write(path: Path, with_contiguous: bool = True) -> Path:
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  w, b = weights()
  inputs = [helper.make_tensor_value_info(n, TensorProto.UINT8 if n.endswith('img') else TensorProto.FLOAT16, s)
            for n, s in SHAPES.items()]
  nodes = [
    helper.make_node('Cast', ['img'], ['img_f'], to=TensorProto.FLOAT16),
    helper.make_node('Cast', ['big_img'], ['big_img_f'], to=TensorProto.FLOAT16),
    helper.make_node('Concat', ['img_f', 'big_img_f'], ['cat'], axis=1),
    helper.make_node('ReduceMean', ['cat'], ['img_mean'], axes=[2, 3], keepdims=0),
    helper.make_node('Flatten', ['desire_pulse'], ['desire_flat'], axis=1),
    helper.make_node('Flatten', ['features_buffer'], ['feat_flat'], axis=1),
    helper.make_node('Concat', ['img_mean', 'desire_flat', 'traffic_convention', 'action_t', 'feat_flat'],
                     ['features'], axis=1),
  ]
  matmul_in = 'features'
  if with_contiguous:
    nodes.append(helper.make_node('Contiguous', ['features'], ['features_c'], domain='org.tinygrad'))
    matmul_in = 'features_c'
  nodes += [
    helper.make_node('MatMul', [matmul_in, 'W'], ['mm']),
    helper.make_node('Add', ['mm', 'B'], ['outputs']),
  ]
  graph = helper.make_graph(
    nodes, 'tiny_driving', inputs,
    [helper.make_tensor_value_info('outputs', TensorProto.FLOAT16, (1, N_OUT))],
    initializer=[numpy_helper.from_array(w, 'W'), numpy_helper.from_array(b, 'B')])
  opsets = [helper.make_opsetid('', 17)]
  if with_contiguous:
    opsets.append(helper.make_opsetid('org.tinygrad', 1))
  model = helper.make_model(graph, opset_imports=opsets)
  model.ir_version = 8
  model.metadata_props.add(key='output_slices',
                           value=codecs.encode(pickle.dumps(SLICES), 'base64').decode())
  model.metadata_props.add(key='model_checkpoint', value='tiny-test')
  onnx.save(model, str(path))
  return path


def reference(inputs: dict[str, np.ndarray]) -> np.ndarray:
  """What the graph computes, in float32."""
  w, b = weights()
  img = np.concatenate([inputs['img'], inputs['big_img']], axis=1).astype(np.float32)
  parts = [img.mean(axis=(2, 3)),
           inputs['desire_pulse'].astype(np.float32).reshape(1, -1),
           inputs['traffic_convention'].astype(np.float32).reshape(1, -1),
           inputs['action_t'].astype(np.float32).reshape(1, -1),
           inputs['features_buffer'].astype(np.float32).reshape(1, -1)]
  feats = np.concatenate(parts, axis=1)
  return (feats @ w.astype(np.float32) + b.astype(np.float32)).reshape(-1)


def random_inputs(seed: int = 0) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  out = {}
  for name, shape in SHAPES.items():
    if name.endswith('img'):
      out[name] = rng.integers(0, 256, shape, dtype=np.uint8)
    else:
      out[name] = (rng.standard_normal(shape) * 0.5).astype(np.float16)
  return out
