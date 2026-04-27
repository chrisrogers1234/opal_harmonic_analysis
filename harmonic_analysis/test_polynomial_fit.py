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
        self.fitter.dimension = dimension
        self.fitter.polynomial_order = order
        polynomial_vector = self.fitter.make_polynomial_vector([[0]*dimension])
        n_poly_coefficients = polynomial_vector.shape[1]
        n_points = n_poly_coefficients*2
        x_in = numpy.random.uniform(-1, 1, [n_points, dimension])
        poly_coefficients = numpy.random.uniform(-1, 1, n_poly_coefficients)
        self.fitter.verbose = 0
        self.fitter.polynomial_order = order
        self.fitter.dimension = dimension
        poly_vectors = self.fitter.make_polynomial_vector(x_in)
        y_out = poly_vectors*poly_coefficients
        y_out = numpy.sum(y_out, axis=1)
        for algorithm in ["linear_least_squares", "differential_evolution", ][0:1]:
            self.fitter.algorithm = algorithm
            test_coeffs = self.fitter.least_squares_fit(x_in, y_out, [-10, 10], 1e-6).x
            self.assertEqual(test_coeffs.size, poly_coefficients.size)
            for i in range(n_poly_coefficients):
                self.assertAlmostEqual(test_coeffs[i], poly_coefficients[i])
            #print("FIT TEST", algorithm)
            #print(test_coeffs)
            #print(poly_coefficients)


class TestMultiPolynomialFit(unittest.TestCase):
    def setUp(self):
        self.multifitter = polynomial_fit.MultipolynomialFit()

    def test_least_squares_fit(self):
        dimension = 3
        order = 2
        n_poly_coefficients = 10
        n_points = n_poly_coefficients*2
        x_in = numpy.random.uniform(-1, 1, [n_points, dimension])
        ref_coeffs = numpy.random.uniform(-1, 1, [dimension, n_poly_coefficients])
        y_ref = []

        for i in range(dimension):
            polynomial = polynomial_fit.PolynomialFit()
            polynomial.polynomial_order = order
            polynomial.dimension = dimension
            polynomial.polynomial_coefficients = ref_coeffs[i]
            y_ref.append(polynomial.function(x_in))
        y_ref = numpy.array(y_ref).transpose()

        self.multifitter.dimension = dimension
        self.multifitter.polynomial_order = order
        self.multifitter.verbose = 1

        optimisation_list = self.multifitter.least_squares_fit(x_in, y_ref, [-10, 10], 1e-6)
        test_coeffs = numpy.array([optim.x for optim in optimisation_list])
        self.assertEqual(test_coeffs.shape, ref_coeffs.shape)
        for i in range(dimension):
            for j in range(n_poly_coefficients):
                self.assertAlmostEqual(test_coeffs[i, j], ref_coeffs[i, j])




if __name__ == "__main__":
    unittest.main()