"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

TensorRT's ONNX parser rejects UINT8 graph inputs outright, so the head of the
model has to be retyped before it will build. Exporters disagree about where
the cast goes, and the patcher has to cope with either.
"""
from __future__ import annotations

import pytest

onnx = pytest.importorskip('onnx')

from onnx import TensorProto, helper  # noqa: E402

from jetlink.onnx_patch import needs_patch, patch_uint8_inputs  # noqa: E402

SHAPE = [1, 12, 128, 256]


def _image_inputs():
  return [helper.make_tensor_value_info(n, TensorProto.UINT8, SHAPE) for n in ('img', 'big_img')]


def _model(nodes, value_info=()):
  graph = helper.make_graph(nodes, 'test', _image_inputs(),
                            [helper.make_tensor_value_info('out', TensorProto.FLOAT16, SHAPE)],
                            value_info=list(value_info))
  return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20)])


def _cast_after_concat():
  """comma's shape: join the uint8 inputs, then cast the result."""
  return _model([
    helper.make_node('Concat', ['img', 'big_img'], ['cat'], axis=1),
    helper.make_node('Cast', ['cat'], ['cast_out'], to=TensorProto.FLOAT16),
    helper.make_node('Identity', ['cast_out'], ['out']),
  ], value_info=[helper.make_tensor_value_info('cat', TensorProto.UINT8, SHAPE)])


def _cast_per_input():
  """sunnypilot's shape: cast each input, then join."""
  return _model([
    helper.make_node('Cast', ['img'], ['a'], to=TensorProto.FLOAT16),
    helper.make_node('Cast', ['big_img'], ['b'], to=TensorProto.FLOAT16),
    helper.make_node('Concat', ['a', 'b'], ['cat'], axis=1),
    helper.make_node('Identity', ['cat'], ['out']),
  ])


@pytest.mark.parametrize(("name", "build"), [
  ('cast after concat', _cast_after_concat),
  ('cast per input', _cast_per_input),
])
class TestBothGraphShapes:
  def test_is_recognised_as_needing_a_patch(self, name, build):
    assert needs_patch(build())

  def test_inputs_come_out_fp16(self, name, build):
    g = patch_uint8_inputs(build()).graph
    assert all(vi.type.tensor_type.elem_type == TensorProto.FLOAT16 for vi in g.input)

  def test_the_casts_are_gone(self, name, build):
    g = patch_uint8_inputs(build()).graph
    assert not [n for n in g.node if n.op_type == 'Cast']

  def test_nothing_still_reads_a_dropped_output(self, name, build):
    g = patch_uint8_inputs(build()).graph
    produced = {o for n in g.node for o in n.output}
    inputs = {vi.name for vi in g.input}
    for n in g.node:
      for i in n.input:
        assert i in produced or i in inputs, f"{n.op_type} reads dangling {i!r}"

  def test_the_result_is_a_valid_model(self, name, build):
    # A stale uint8 value_info is what onnxruntime refuses to load, and the
    # checker is the cheapest thing that notices.
    onnx.checker.check_model(patch_uint8_inputs(build()))

  def test_patching_twice_is_refused_rather_than_silent(self, name, build):
    once = patch_uint8_inputs(build())
    assert not needs_patch(once)
    with pytest.raises(ValueError):
      patch_uint8_inputs(once)


def test_a_model_without_image_inputs_says_so():
  graph = helper.make_graph([helper.make_node('Identity', ['x'], ['out'])], 'g',
                            [helper.make_tensor_value_info('x', TensorProto.UINT8, SHAPE)],
                            [helper.make_tensor_value_info('out', TensorProto.UINT8, SHAPE)])
  with pytest.raises(ValueError, match="none of"):
    patch_uint8_inputs(helper.make_model(graph))


def test_a_cast_to_the_wrong_type_is_refused():
  model = _model([
    helper.make_node('Cast', ['img'], ['a'], to=TensorProto.FLOAT),
    helper.make_node('Cast', ['big_img'], ['b'], to=TensorProto.FLOAT),
    helper.make_node('Concat', ['a', 'b'], ['out'], axis=1),
  ])
  with pytest.raises(ValueError, match="expected FLOAT16"):
    patch_uint8_inputs(model)
