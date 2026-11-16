import unittest
from types import SimpleNamespace

import numpy as np

from wrapper.caps.s3_ov_llm import StatefulGPU


class CausalRequest:
    def __init__(self, projected):
        self.projected, self.value, self.lengths = projected, 0., []

    def infer(self, inputs, share_inputs=False):
        ids = inputs['input_ids']
        self.lengths.append(ids.shape[1])
        values = np.where(inputs['embedding_override_mask'][0, :, 0],
                          inputs['embedding_override'][0, :, 0], ids[0])
        hidden = self.value + np.cumsum(values)
        self.value = float(hidden[-1])
        hidden = hidden[None, :, None]
        self.result = {'logits': hidden[:, -1:], 'final_hidden': hidden if self.projected else hidden[:, -1:]}
        if self.projected:
            self.result['projected_hidden'] = hidden * 2

    def get_tensor(self, name):
        return SimpleNamespace(data=self.result[name])

    def query_state(self):
        return [SimpleNamespace(reset=lambda: setattr(self, 'value', 0.))]


def runtime(projected=True):
    value = StatefulGPU.__new__(StatefulGPU)
    value.compiled = SimpleNamespace(create_infer_request=lambda: CausalRequest(projected))
    value.request, value.width, value.position = value.compiled.create_infer_request(), 1, 0
    value.inputs = {'input_ids', 'attention_mask', 'position_ids', 'beam_idx',
                    'embedding_override', 'embedding_override_mask'}
    value.outputs = {name: name for name in ['logits', 'final_hidden'] + (['projected_hidden'] if projected else [])}
    return value


class PrefillTests(unittest.TestCase):
    def test_split_preserves_every_hidden_and_continuation_state(self):
        engine = runtime()
        ids = np.arange(1, 204, dtype=np.int64)[None]
        embeds = np.full((1, 203, 1), 7., dtype=np.float32)
        mask = (ids % 3 == 0)[:, :, None]
        result = engine.infer(ids, embeddings=embeds, mask=mask)
        expected = np.cumsum(np.where(mask[0, :, 0], 7., ids[0]))
        np.testing.assert_array_equal(result['final_hidden'][0, :, 0], expected)
        np.testing.assert_array_equal(result['projected_hidden'][0, :, 0], expected * 2)
        self.assertEqual(engine.position, 203)
        self.assertEqual(engine.request.lengths, [1] * 203)
        self.assertEqual(float(engine.infer([5])['logits'][0, 0, 0]), expected[-1] + 5)
        self.assertEqual(engine.position, 204)
        engine.reset()
        self.assertEqual(float(engine.infer([5])['logits'][0, 0, 0]), 5.)

    def test_reset_isolates_a_second_source_and_preserves_all_positions(self):
        engine = runtime()
        engine.infer(np.arange(1, 74))
        previous = engine.request
        engine.reset()
        ids = np.arange(91, 122)
        result = engine.infer(ids)
        self.assertIsNot(engine.request, previous)
        np.testing.assert_array_equal(result['projected_hidden'][0, :, 0], np.cumsum(ids) * 2)
        self.assertEqual(engine.position, len(ids))

    def test_talker_accepts_unbatched_embedding_contract(self):
        engine = runtime(False)
        values = np.arange(1, 131, dtype=np.float32).reshape(130, 1)
        result = engine.infer(np.zeros(130, dtype=np.int64), embeddings=values)
        self.assertEqual(float(result['final_hidden'][0, 0, 0]), float(values.sum()))
        self.assertEqual(engine.position, 130)

    def test_talker_keeps_only_last_hidden_and_all_context(self):
        engine = runtime(False)
        result = engine.infer(list(range(1, 131)))
        self.assertEqual(result['final_hidden'].shape, (1, 1, 1))
        self.assertEqual(float(result['final_hidden'][0, 0, 0]), sum(range(1, 131)))
        self.assertEqual(engine.position, 130)
