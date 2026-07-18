import unittest

import numpy as np

from tools.fuse_all_float import (
    all_float_vertex,
    hotspot_weights,
    nlerp_quaternion,
    validate_pair,
    weighted_all_float_vertex,
)


def vertex(count=2):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("f_rest_0", "f4"), ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("label", "i4"),
    ]
    value = np.zeros(count, dtype=dtype)
    value["rot_0"] = 1.0
    value["label"] = np.arange(count)
    return value


class FusionTests(unittest.TestCase):
    def test_all_float_endpoints_are_byte_exact(self):
        rgb = vertex()
        thermal = rgb.copy()
        thermal["f_dc_0"] = [1.0, 2.0]
        self.assertTrue(np.array_equal(all_float_vertex(rgb, thermal, 0.0), rgb))
        self.assertTrue(np.array_equal(all_float_vertex(rgb, thermal, 1.0), thermal))

    def test_q_negative_q_does_not_collapse(self):
        left = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        right = -left
        result = nlerp_quaternion(left, right, np.asarray([0.5]))
        np.testing.assert_allclose(result, left)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_pair_requires_shared_geometry(self):
        rgb = vertex()
        thermal = rgb.copy()
        validate_pair(rgb, thermal, True)
        thermal["x"][0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "shared-geometry"):
            validate_pair(rgb, thermal, True)

    def test_hotspot_weight_keeps_cold_rows_closer_to_rgb(self):
        rgb = vertex()
        thermal = rgb.copy()
        thermal["f_dc_0"] = 1.0
        weight, gate = hotspot_weights(
            np.asarray([0.0, 10.0]), threshold_c=5.0, softness_c=0.5, strength=1.0
        )
        output = weighted_all_float_vertex(rgb, thermal, weight)
        self.assertLess(float(output["f_dc_0"][0]), 0.001)
        self.assertGreater(float(output["f_dc_0"][1]), 0.999)
        self.assertLess(float(gate[0]), float(gate[1]))


if __name__ == "__main__":
    unittest.main()
