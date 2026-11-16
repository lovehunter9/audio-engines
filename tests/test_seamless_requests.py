import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


source = Path(__file__).parents[1] / "wrapper/caps/seamless_ov.py"
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Graph")
namespace = {"np": np, "_numpy": np.asarray}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
Graph = namespace["Graph"]


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data).copy()


class Request:
    def __init__(self, stateful):
        self.stateful, self.inputs = stateful, {}
        self.outputs = [Tensor(np.zeros(1)) for _ in range(2 if stateful else 1)]

    def set_input_tensor(self, index, tensor):
        self.inputs[index] = tensor

    def infer(self):
        for tensor in self.outputs:
            tensor.data.fill(-999)
        previous = self.inputs[1].data if self.stateful else 0
        value = self.inputs[0].data + previous
        self.outputs[0].data[:] = value * 2
        if self.stateful:
            self.outputs[1].data[:] = value

    def get_output_tensor(self, index):
        return self.outputs[index]


class Compiled:
    def __init__(self, stateful):
        self.stateful = stateful
        self.inputs = [None] * (2 if stateful else 1)
        self.outputs = [None] * (2 if stateful else 1)
        self.created = 0

    def create_infer_request(self):
        self.created += 1
        return Request(self.stateful)


def graph(stateful):
    value = Graph.__new__(Graph)
    value.compiled = Compiled(stateful)
    value.requests, value.request_index = [], 0
    return value


class SeamlessRequestTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {"openvino": SimpleNamespace(Tensor=Tensor)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_retained_cache_is_not_overwritten_by_its_next_inference(self):
        engine, cache = graph(True), Tensor(np.zeros(1))
        features = []
        for token, expected in [(1, 1), (2, 3), (3, 6), (4, 10)]:
            feature, cache = engine.infer([np.array([token]), cache], retain=1)
            np.testing.assert_array_equal(cache.data, [expected])
            features.append(feature)
        self.assertEqual(engine.compiled.created, 2)
        np.testing.assert_array_equal(np.concatenate(features), [2, 6, 12, 20])

    def test_stateless_outputs_survive_later_request_reuse(self):
        engine = graph(False)
        outputs = [engine.infer([np.array([value])])[0] for value in [3, 1, 4, 2]]
        self.assertEqual(engine.compiled.created, 1)
        np.testing.assert_array_equal(np.concatenate(outputs), [6, 2, 8, 4])


if __name__ == "__main__":
    unittest.main()
