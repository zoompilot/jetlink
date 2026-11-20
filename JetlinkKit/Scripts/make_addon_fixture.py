"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Writes Tests/JetlinkServerTests/Fixtures/tiny_addon.onnx: an add-on
shaped graph, one float [1, 3, 4, 4] image in, two outputs whose names sort
the other way round from their order in the graph. "b_mean" is the image's
per-channel mean, "a_scaled" is the image times two, flattened.
"""
from pathlib import Path

from onnx import TensorProto, helper, save

OUT = Path(__file__).resolve().parents[1] / 'Tests/JetlinkServerTests/Fixtures/tiny_addon.onnx'


def main() -> None:
  image = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 4, 4])
  mean = helper.make_tensor_value_info('b_mean', TensorProto.FLOAT, [1, 3])
  scaled = helper.make_tensor_value_info('a_scaled', TensorProto.FLOAT, [1, 48])
  two = helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])
  nodes = [
    helper.make_node('ReduceMean', ['images'], ['b_mean'], axes=[2, 3], keepdims=0),
    helper.make_node('Mul', ['images', 'two'], ['doubled']),
    helper.make_node('Flatten', ['doubled'], ['a_scaled'], axis=1),
  ]
  graph = helper.make_graph(nodes, 'tiny_addon', [image], [mean, scaled], initializer=[two])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], producer_name='jetlink-tests')
  model.ir_version = 8
  save(model, OUT)


if __name__ == '__main__':
  main()
