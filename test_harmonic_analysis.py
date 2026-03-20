import unittest

import numpy
import matplotlib.pyplot

import harmonic_analysis


class FieldMockup():
    """
    Arbitrary multipole. Field is given by [cite wikipedia]
        Bz + iBx = C_n (x - i z)^{n-1}
    where C_n is complex. In OPAL-CYCL y is (initially) forwards, z is up
    """
    def __init__(self):
        # apply formula
        # B = C + C_i u^i + C_ij u^i u^j + ...
        # where u^i are components of unit vector
        self.rotation = numpy.identity(3)
        self.centre = numpy.array([0, 0, 0])
        self.c_n = []

    def get_field_value(self, x, y, z, t):
        pos = numpy.array([x, y, z]) - self.centre
        pos = numpy.dot(self.rotation, pos)
        bx, by, bz = 0.0, 0.0, 0.0
        for n, c in enumerate(self.c_n):
            bfield = c * (pos[0] + pos[2]*1j)**n
            bx += numpy.imag(bfield)
            bz += numpy.real(bfield)
        field = (False, bx, by, bz, 0.0, 0.0, 0.0)
        return field

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
        centre, coordinates = self.analysis.build_rotated_vectors(psv, h, v)
        self.assertEqual(len(coordinates), self.default_config["harmonic"])
        for i in range(3):
            self.assertEqual(centre[i], [1, 2, 3][i])
        for i, c in enumerate(coordinates):
            c = (c - numpy.array([1, 2, 3]))/self.default_config["delta"]
            self.assertAlmostEqual(numpy.dot(c, l), 0.0)
            self.assertAlmostEqual(numpy.dot(c, h), numpy.cos(2*numpy.pi*i/8))
            self.assertAlmostEqual(numpy.dot(c, v), numpy.sin(2*numpy.pi*i/8))

    def test_calculate_field(self):
        psv = {"x":0, "y":0, "z":0, "xp":0, "yp":1, "zp":0}
        h, v, l = self.analysis.get_coordinate_system(psv)
        centre, coordinates = self.analysis.build_rotated_vectors(psv, h, v)
        bh, bv, bl = self.analysis.calculate_field(h, v, l, coordinates)
        for b_array in bh, bv, bl:
            self.assertEqual(len(b_array), 8)
            for b in b_array:
                self.assertEqual(b, 0)
        self.analysis.field.c_n = [1.0]
        bh, bv, bl = self.analysis.calculate_field(h, v, l, coordinates)
        for i, b in enumerate(bv):
            self.assertEqual(b, 1.0)
        for i, b in enumerate(bh):
            self.assertEqual(b, 0)
        for i, b in enumerate(bl):
            self.assertEqual(b, 0)

        self.analysis.field.c_n = [0, 1.0]
        bh, bv, bl = self.analysis.calculate_field(h, v, l, coordinates)
        for i, b in enumerate(bv):
            dx = coordinates[i][0] # vertical field = horizontal position*gradient
            self.assertEqual(b, dx)
        for i, b in enumerate(bh):
            dz = coordinates[i][2] # horizontal field = vertical position*gradient
            self.assertEqual(b, dz)
        for i, b in enumerate(bl):
            self.assertEqual(b, 0)

    def test_calculate_field_r(self):
        self.analysis.field.c_n = [1.0]
        psv = {"x":0, "y":0, "z":0, "xp":0, "yp":1, "zp":0}
        h, v, l = self.analysis.get_coordinate_system(psv)
        centre, coordinates = self.analysis.build_rotated_vectors(psv, h, v)
        br = self.analysis.calculate_field_r(centre, coordinates)
        self.assertEqual(len(br), len(coordinates))
        for i, b in enumerate(br):
            c = coordinates[i]
            self.assertEqual(b, numpy.dot(c, v)/self.default_config["delta"])

    def test_analyse_one_step(self):
        centre = [7, 8, 9]
        psv = {"x":centre[0], "y":centre[1], "z":centre[2], "xp":0, "yp":1, "zp":0, "step":-1}
        self.default_config["harmonic"] = 128
        self.default_config["delta"] = 3
        #c_n = [1+4j, 2, 3]
        c_n = []
        for i in range(3):
            real_c = numpy.random.uniform(-10, 10)
            imag_c = numpy.random.uniform(-10, 10)
            c_n.append(real_c+imag_c*1j)
        self.analysis.field.c_n = c_n
        self.analysis.field.centre = numpy.array(centre)
        a_fft = self.analysis.analyse_one_step(psv, self.do_plots)
        self.analysis = harmonic_analysis.HarmonicAnalysis(self.default_config)
        self.analysis.field = FieldMockup()
        for i, c_ni in enumerate(c_n):
            self.assertAlmostEqual(c_ni, a_fft[i+1])
        if self.do_plots:
            matplotlib.pyplot.show(block=False)
            input()

    do_plots = False

if __name__ == "__main__":
    TestHarmonicAnalysis.do_plots = True
    unittest.main()
