import unittest

import numpy

import harmonic_analysis


class FieldMockup():
    def __init__(self):
        pass

    def get_field_value(x, y, z, t):
        return [False]+[0.0]*3+[0.0]*3

class TestHarmonicAnalysis(unittest.TestCase):
    def setUp(self):
        self.default_config = {
            "harmonic":8,
            "delta":2,
        }
        self.analysis = harmonic_analysis.HarmonicAnalysis(self.default_config)
        self.analysis.field = FieldMockup()

    def test_get_x_vector(self):
        psv = {"x":0.1, "y":0.2, "z":0.3}
        test = self.analysis.get_x_vector(psv)
        for i in range(3):
            self.assertAlmostEqual(test[i], [0.1, 0.2, 0.3][i])

    def test_get_p_vector(self):
        psv = {"xp":1, "yp":2, "zp":3}
        test = self.analysis.get_p_vector(psv)
        for i in range(3):
            self.assertAlmostEqual(test[i], [1/14**0.5, 2/14**0.5, 3/14**0.5][i])

    def test_get_coordinate_system(self):
        psv = {"xp":5, "yp":0, "zp":0}
        h, v, l = self.analysis.get_coordinate_system(psv)
        for i in range(3):
            self.assertAlmostEqual(h[i], [0, -1, 0][i])
            self.assertAlmostEqual(v[i], [0, 0, 1][i])
            self.assertAlmostEqual(l[i], [1, 0, 0][i])
        # choose a psv in an odd direction
        psv = {"xp":5, "yp":5, "zp":5}
        h, v, l = self.analysis.get_coordinate_system(psv)
        # longitudinal 1, 1, 1 normalised
        for i in range(3):
            self.assertAlmostEqual(l[i]*3**0.5, [1, 1, 1][i])
        # all vectors should be length 1
        for vec in h, v, l:
            self.assertAlmostEqual(numpy.linalg.norm(vec), 1.0)
        # all vectors should be perpendicular
        self.assertAlmostEqual(numpy.dot(h, v), 0.0)
        self.assertAlmostEqual(numpy.dot(h, l), 0.0)
        self.assertAlmostEqual(numpy.dot(v, l), 0.0)

    def test_build_rotated_vectors(self):
        psv = {"x":1, "y":2, "z":3, "xp":5, "yp":0, "zp":0}
        h, v, l = self.analysis.get_coordinate_system(psv)
        coordinates = self.analysis.build_rotated_vectors(psv, h, v)
        self.assertEqual(len(coordinates), self.default_config["harmonic"])
        for i, c in enumerate(coordinates):
            c = (c - numpy.array([1, 2, 3]))/self.default_config["delta"]
            self.assertAlmostEqual(numpy.dot(c, l), 0.0)
            self.assertAlmostEqual(numpy.dot(c, h), numpy.cos(2*numpy.pi*i/8))
            self.assertAlmostEqual(numpy.dot(c, v), numpy.sin(2*numpy.pi*i/8))

    


if __name__ == "__main__":
    unittest.main()