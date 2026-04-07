import polynomial_fit

import numpy

import unittest

class TestPolynomialFit(unittest.TestCase):
    def setUp(self):
        self.fitter = polynomial_fit.PolynomialFit()

    def test_make_polynomial_indices(self):
        self.fitter.dimension = 3
        self.fitter.polynomial_order = 2
        test_indices = self.fitter.make_polynomial_indices()
        ref_indices = [
            [],
            [0], [1], [2],
            [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2,2]
        ]
        self.assertEqual(test_indices, ref_indices)

    def test_make_variable_vector(self):
        for order, ref_vec in [(-1, [[1.0]]), (0, [[1.0]]), (1, [[1.0, 2.0, 3.0, 4.0]]),
                (2, [[1, 2, 3, 4, 4, 6, 8, 9, 12, 16]])]:
            self.fitter.dimension = 3
            self.fitter.polynomial_order = order
            ref_vec = numpy.array(ref_vec)
            polynomial_vector = self.fitter.make_polynomial_vector([[2, 3, 4]])
            self.assertTrue(numpy.array_equal(polynomial_vector, ref_vec))
        self.fitter.polynomial_order = 2
        polynomial_vector = self.fitter.make_polynomial_vector([[2, 3, 4]]*3)
        ref_vec = numpy.array([[1, 2, 3, 4, 4, 6, 8, 9, 12, 16]]*3)
        self.assertTrue(numpy.array_equal(polynomial_vector, ref_vec))

    def test_least_squares_fit(self):
        dimension = 3
        order = 2
        n_poly_coefficients = 10
        n_points = n_poly_coefficients*2
        x_in = numpy.random.uniform(-1, 1, [n_points, dimension])
        poly_coefficients = numpy.random.uniform(-1, 1, n_poly_coefficients)
        self.fitter.polynomial_order = order
        self.fitter.dimension = dimension
        poly_vectors = self.fitter.make_polynomial_vector(x_in)
        self.fitter.polynomial_order = 0 # check this gets filled by least_squares_fit
        self.fitter.dimension = 0 # check this gets filled by least_squares_fit
        y_out = poly_vectors*poly_coefficients
        y_out = numpy.sum(y_out, axis=1)
        test_coeffs = self.fitter.least_squares_fit(x_in, y_out, order, [-10, 10], 1e-6).x
        self.assertEqual(test_coeffs.size, poly_coefficients.size)
        for i in range(n_poly_coefficients):
            self.assertAlmostEqual(test_coeffs[i], poly_coefficients[i])






if __name__ == "__main__":
    unittest.main()