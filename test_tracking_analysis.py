import tracking_analysis

import numpy

import unittest

class TestTrackingAnalysis(unittest.TestCase):
    def setUp(self):
        self.analysis = tracking_analysis.TrackingAnalysis()

    def test_make_polynomial_indices(self):
        test_indices = self.analysis.make_polynomial_indices(3, 2)
        ref_indices = [
            [],
            [0], [1], [2],
            [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2,2]
        ]
        self.assertEqual(test_indices, ref_indices)

    def test_make_variable_vector(self):
        for order, ref_vec in [(-1, [[1.0]]), (0, [[1.0]]), (1, [[1.0, 2.0, 3.0, 4.0]]),
                (2, [[1, 2, 3, 4, 4, 6, 8, 9, 12, 16]])]:
            ref_vec = numpy.array(ref_vec)
            polynomial_vector = self.analysis.make_polynomial_vector([[2, 3, 4]],
                                                                     order)
            self.assertTrue(numpy.array_equal(polynomial_vector, ref_vec))
        polynomial_vector = self.analysis.make_polynomial_vector([[2, 3, 4]]*3,
                                                                     order)
        ref_vec = numpy.array([[1, 2, 3, 4, 4, 6, 8, 9, 12, 16]]*3)
        self.assertTrue(numpy.array_equal(polynomial_vector, ref_vec))

    def test_least_squares_fit(self):
        dimension = 3
        order = 2
        n_poly_coefficients = 10
        n_points = n_poly_coefficients*2
        x_in = numpy.random.uniform(-1, 1, [n_points, dimension])
        poly_coefficients = numpy.random.uniform(-1, 1, n_poly_coefficients)
        poly_vectors = self.analysis.make_polynomial_vector(x_in, order)
        y_out = poly_vectors*poly_coefficients
        y_out = numpy.sum(y_out, axis=1)
        test_coeffs = self.analysis.least_squares_fit(x_in, y_out, order, [-10, 10], 1e-6).x
        self.assertEqual(test_coeffs.size, poly_coefficients.size)
        for i in range(n_poly_coefficients):
            self.assertAlmostEqual(test_coeffs[i], poly_coefficients[i])
        #print(test_coeffs)
        #print(poly_coefficients)





if __name__ == "__main__":
    unittest.main()