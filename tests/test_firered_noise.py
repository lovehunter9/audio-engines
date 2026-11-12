import unittest

import numpy as np

from wrapper.caps.firered_noise import PatchNoise


FIXTURES = [{'seed': 2, 'step': 0, 'values': [0.02234119549393654, -0.5777389407157898, 0.7052561044692993, -0.40886709094047546, 1.288116216659546, 1.2293621301651, 1.2962061166763306]}, {'seed': 2, 'step': 1, 'values': [2.266852855682373, 1.347745656967163, -1.4438210725784302, 1.9884870052337646, -0.5623780488967896, -2.2646191120147705, 0.9388427138328552]}, {'seed': 2, 'step': 17, 'values': [-1.8594069480895996, -1.0734635591506958, 0.8185129761695862, 0.11900077015161514, -0.3586404025554657, 0.3494437634944916, -0.20818272233009338]}, {'seed': 1234, 'step': 0, 'values': [-1.6164904832839966, 0.568455159664154, -0.5102248191833496, 1.0653820037841797, -0.300402969121933, -0.16792136430740356, -1.7603728771209717]}, {'seed': 1234, 'step': 1, 'values': [0.6330825686454773, 1.6358213424682617, -0.34586283564567566, -1.6059002876281738, 1.1443605422973633, 0.06653439998626709, -1.0147119760513306]}, {'seed': 1234, 'step': 17, 'values': [-0.9334203004837036, -1.1406055688858032, -1.2331136465072632, -1.1119163036346436, -0.0940166637301445, -0.6790551543235779, -1.101250171661377]}, {'seed': 4294967297, 'step': 0, 'values': [0.8548939824104309, -0.11420174688100815, -1.0717551708221436, 0.5221496224403381, -0.6927298307418823, -0.7221677303314209, -1.0305591821670532]}, {'seed': 4294967297, 'step': 1, 'values': [-0.03069646842777729, 0.7400043606758118, 0.04820120707154274, 0.7864040732383728, -0.29971763491630554, 0.3757341802120209, -0.4879806339740753]}, {'seed': 4294967297, 'step': 17, 'values': [-0.05091742426156998, -1.6099669933319092, -0.4297949969768524, -0.3852778971195221, 0.2514672875404358, -0.3343803882598877, -0.5777391791343689]}]


class PatchNoiseTest(unittest.TestCase):
    def test_cuda_reference_multiple_seeds_and_steps(self):
        for seed in sorted({row["seed"] for row in FIXTURES}):
            sampler = PatchNoise(seed)
            for step in range(18):
                noise = sampler.normal(4, 64)
                row = next((r for r in FIXTURES if r["seed"] == seed and r["step"] == step), None)
                if row is not None:
                    with self.subTest(seed=seed, step=step):
                        np.testing.assert_allclose(noise.reshape(-1)[[0, 1, 2, 7, 31, 127, 255]],
                                                   row["values"], rtol=0, atol=2e-6)

    def test_request_reset_is_deterministic_and_steps_are_distinct(self):
        sampler = PatchNoise(1234)
        first = sampler.normal(4, 64)
        second = sampler.normal(4, 64)
        np.testing.assert_array_equal(first, PatchNoise(1234).normal(4, 64))
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.dtype("float32"))
        self.assertEqual(first.shape, (1, 4, 64))

    def test_unsupported_shape_is_not_silently_sampled_differently(self):
        for size in (0, 257):
            with self.assertRaises(ValueError):
                PatchNoise(1234).normal(1, size)


if __name__ == "__main__":
    unittest.main()
