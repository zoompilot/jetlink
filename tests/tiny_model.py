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


# -- the stateful layout (openpilot #38916) ----------------------------------
# The same structure as Cinque Terre V3, shrunk: the newest frame pair comes in
# as new_img and joins a uint8 frame queue the graph hands back; the model
# reads frames 0 and 4 of it through a Gather/Slice/Reshape chain that ends in
# the one fp16 Cast; the desire pulse joins its own queue; the hidden state is
# sliced out of the output and pushed onto a feature queue, never leaving the
# graph.

STATEFUL_SHAPES = {
  'new_img': (2, 6, 8, 16),
  'desire': (8,),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'state_img_q': (2, 5, 6, 8, 16),
  'state_desire_q': (6, 1, 8),
  'state_feat_q': (4, 1, 16),
}
STATE_PAIRS = {n: f'next_{n}' for n in STATEFUL_SHAPES if n.startswith('state_')}
STATEFUL_SLICES = {'plan': slice(0, 16), 'lead_prob': slice(16, 19), 'hidden_state': slice(32, 48)}
STATEFUL_FEATURES = 24 + 6 * 8 + 2 + 2 + 4 * 16
_BIG = 2 ** 62


def stateful_weights(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  w = (rng.standard_normal((STATEFUL_FEATURES, N_OUT)) * 0.05).astype(np.float32)
  b = (rng.standard_normal(N_OUT) * 0.1).astype(np.float32)
  return w, b


def write_stateful(path: Path) -> Path:
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  w, b = stateful_weights()
  types = {'new_img': TensorProto.UINT8, 'state_img_q': TensorProto.UINT8}
  inputs = [helper.make_tensor_value_info(n, types.get(n, TensorProto.FLOAT), s) for n, s in STATEFUL_SHAPES.items()]
  const = {
    'i0': np.array(0, np.int64), 'i1': np.array(1, np.int64),
    'ax0': np.array([0], np.int64), 'ax1': np.array([1], np.int64),
    'one': np.array([1], np.int64), 'zero': np.array([0], np.int64), 'end': np.array([_BIG], np.int64),
    'four': np.array([4], np.int64), 'h0': np.array([32], np.int64), 'h1': np.array([48], np.int64),
    'cam_shape': np.array([1, 12, 8, 16], np.int64), 'desire_row': np.array([1, 1, 8], np.int64),
    'desire_flat': np.array([1, 48], np.int64), 'feat_flat': np.array([1, 64], np.int64),
    'hidden_row': np.array([1, 1, 16], np.int64), 'scale': np.array(1 / 255, np.float32),
  }
  nodes = [
    # the frame queue, uint8 all the way to the head Cast
    helper.make_node('Unsqueeze', ['new_img', 'ax1'], ['unsqueeze']),
    helper.make_node('Slice', ['state_img_q', 'one', 'end', 'ax1'], ['img_tail']),
    helper.make_node('Concat', ['img_tail', 'unsqueeze'], ['next_state_img_q'], axis=1),
    helper.make_node('Gather', ['next_state_img_q', 'i0'], ['road'], axis=0),
    helper.make_node('Gather', ['next_state_img_q', 'i1'], ['wide'], axis=0),
    helper.make_node('Slice', ['road', 'zero', 'end', 'ax0', 'four'], ['road_pair']),
    helper.make_node('Slice', ['wide', 'zero', 'end', 'ax0', 'four'], ['wide_pair']),
    helper.make_node('Reshape', ['road_pair', 'cam_shape'], ['road_img']),
    helper.make_node('Reshape', ['wide_pair', 'cam_shape'], ['wide_img']),
    helper.make_node('Concat', ['road_img', 'wide_img'], ['imgs'], axis=1),
    helper.make_node('Cast', ['imgs'], ['imgs_f16'], to=TensorProto.FLOAT16),
    helper.make_node('Cast', ['imgs_f16'], ['imgs_f32'], to=TensorProto.FLOAT),
    helper.make_node('ReduceMean', ['imgs_f32'], ['img_sum'], axes=[2, 3], keepdims=0),
    helper.make_node('Mul', ['img_sum', 'scale'], ['img_mean']),
    # the desire queue
    helper.make_node('Reshape', ['desire', 'desire_row'], ['desire_new']),
    helper.make_node('Slice', ['state_desire_q', 'one', 'end', 'ax0'], ['desire_tail']),
    helper.make_node('Concat', ['desire_tail', 'desire_new'], ['next_state_desire_q'], axis=0),
    helper.make_node('Reshape', ['next_state_desire_q', 'desire_flat'], ['desire_in']),
    # the policy, reading last frame's features
    helper.make_node('Reshape', ['state_feat_q', 'feat_flat'], ['feat_in']),
    helper.make_node('Concat', ['img_mean', 'desire_in', 'traffic_convention', 'action_t', 'feat_in'],
                     ['features'], axis=1),
    helper.make_node('MatMul', ['features', 'W'], ['mm']),
    helper.make_node('Add', ['mm', 'B'], ['outputs']),
    # the hidden state goes onto the feature queue
    helper.make_node('Slice', ['outputs', 'h0', 'h1', 'ax1'], ['hidden']),
    helper.make_node('Reshape', ['hidden', 'hidden_row'], ['hidden_new']),
    helper.make_node('Slice', ['state_feat_q', 'one', 'end', 'ax0'], ['feat_tail']),
    helper.make_node('Concat', ['feat_tail', 'hidden_new'], ['next_state_feat_q'], axis=0),
  ]
  outputs = [helper.make_tensor_value_info('outputs', TensorProto.FLOAT, (1, N_OUT))]
  outputs += [helper.make_tensor_value_info(nxt, types.get(n, TensorProto.FLOAT), STATEFUL_SHAPES[n])
              for n, nxt in STATE_PAIRS.items()]
  graph = helper.make_graph(nodes, 'tiny_stateful', inputs, outputs,
                            initializer=[numpy_helper.from_array(w, 'W'), numpy_helper.from_array(b, 'B')]
                            + [numpy_helper.from_array(v, k) for k, v in const.items()])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
  model.ir_version = 8
  model.metadata_props.add(key='output_slices',
                           value=codecs.encode(pickle.dumps(STATEFUL_SLICES), 'base64').decode())
  model.metadata_props.add(key='model_checkpoint', value='tiny-stateful-test')
  onnx.save(model, str(path))
  return path


def empty_state() -> dict[str, np.ndarray]:
  return {n: np.zeros(STATEFUL_SHAPES[n], np.uint8 if n == 'state_img_q' else np.float32) for n in STATE_PAIRS}


def stateful_step(state: dict[str, np.ndarray], new_img: np.ndarray, desire: np.ndarray,
                  traffic_convention: np.ndarray, action_t: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  """One frame of the graph in float32: the output and the next state."""
  w, b = stateful_weights()
  img_q = np.concatenate([state['state_img_q'][:, 1:], new_img[:, None]], axis=1)
  desire_q = np.concatenate([state['state_desire_q'][1:], desire.reshape(1, 1, 8).astype(np.float32)], axis=0)
  imgs = np.concatenate([img_q[c, ::4].reshape(1, 12, 8, 16) for c in (0, 1)], axis=1).astype(np.float32)
  feats = np.concatenate([imgs.mean(axis=(2, 3)) / 255, desire_q.reshape(1, -1),
                          traffic_convention.reshape(1, -1).astype(np.float32),
                          action_t.reshape(1, -1).astype(np.float32),
                          state['state_feat_q'].reshape(1, -1)], axis=1)
  out = feats @ w + b
  feat_q = np.concatenate([state['state_feat_q'][1:], out[:, 32:48].reshape(1, 1, 16)], axis=0)
  return out.reshape(-1), {'state_img_q': img_q, 'state_desire_q': desire_q, 'state_feat_q': feat_q}


def stateful_frames(n: int, seed: int = 0) -> list[dict[str, np.ndarray]]:
  """Per-frame inputs: a frame pair, a desire pulse now and then, the scalars."""
  rng = np.random.default_rng(seed)
  frames = []
  for i in range(n):
    desire = np.zeros(8, np.float32)
    if i % 3 == 1:
      desire[rng.integers(1, 8)] = 1
    frames.append({'new_img': rng.integers(0, 256, STATEFUL_SHAPES['new_img'], dtype=np.uint8),
                   'desire': desire,
                   'traffic_convention': np.array([[1, 0]], np.float32),
                   'action_t': rng.standard_normal((1, 2)).astype(np.float32) * 0.1})
  return frames
