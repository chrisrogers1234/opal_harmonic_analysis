import os
import shutil
import time

import numpy
import h5py
import scipy
import itertools

import pyopal.objects.minimal_runner
import pyopal.elements.multipolet
import pyopal.elements.output_plane


class MultipolynomialFit:
    """
    MultiPolynomialFit calculates and stores coefficients for ND to ND polynomials

    Polynomial is defined as
        y_a = m_a + m_ai x_ai + m_aij x_i x_j + ...
    where x_i are elements of an input vector and y_a are elements of an output
    vector. In the accelerator context, x can be vector of position, momentum,
    etc and y also a vector of position, momentum etc at a different point in
    the accelerator. For accelerator context, we *require* that y is the same
    length as x (although this is just for convenience, not required by maths).

    Member data:
    - one_d_polynomials: List of PolynomialFit objects, one for each item in y
    Other member data follows the definition in PolynomialFit.
    """
    def __init__(self):
        self.one_d_polynomials = []
        self.polynomial_order = 0
        self.dimension = 4
        # control print output
        self.verbose = 1
        self.print_time_step = 1
        self.max_iter = int(1e9)
        # output of most recent fit attempt
        self.optimisation_list = []

    def least_squares_fit(self, x_in, y_out, poly_limits, tolerance):
        """
        Do least squares fit. Each 1D polynomial is fit independently.
        """
        self.one_d_polynomials = []
        if self.dimension != y_out.shape[1]:
            raise ValueError("Output vector lengths did not match dimension")
        self.optimisation_list = []
        for i in range(self.dimension):
            polynomial_1d = PolynomialFit()
            polynomial_1d.polynomial_order = self.polynomial_order
            polynomial_1d.dimension = self.dimension
            polynomial_1d.verbose = self.verbose
            polynomial_1d.print_time_step = self.print_time_step
            polynomial_1d.max_iter = self.max_iter
            y_out_1d = y_out[:,i].transpose()
            print("Fitting", y_out_1d)
            out = polynomial_1d.least_squares_fit(x_in, y_out_1d, poly_limits, tolerance)
            self.optimisation_list.append(out)
            self.one_d_polynomials.append(polynomial_1d)
        return self.optimisation_list

    def function(self, x_in):
        y_out = numpy.array([polynomial.function(x_in) for polynomial in self.one_d_polynomials])



class PolynomialFit:
    """
    Polynomial fit calculates and stores coefficients for ND to 1D polynomials

    Polynomial is defined as
        y = m + m_i x_i + m_ij x_i x_j + ...
    where x_i are elements of an input vector and y is an output scalar. In the
    accelerator context, x can be position, momentum, etc and y is an output
    position or momentum

    Member data:
    - polynomial_coefficients: these are the coefficients like m, m_i, m_ij, ...
    - polynomial_order: the order of the polynomial. < 1 returns [1], 1 is linear,
                        2 is quadratic etc.
    - dimension: the dimension of the input variable
    - verbose: set to 0 to be quiet. > 10 prints everything
    - print_time_step: for longer solves, set the interval between outputs in seconds
    - max_iter: maximum number of iterations during the fit
    """

    def __init__(self):
        # polynomial coefficients are stored here
        self.polynomial_coefficients = []
        # set by least_squares_fit method
        self.polynomial_order = 0
        self.dimension = 4
        # control print output
        self.verbose = 1
        self.print_time_step = 1
        self.max_iter = int(1e9)
        # ephemeral data used during fitting
        self.polynomial_data_tmp = None
        self.coefficients_one_d_tmp = None
        self.iter_tmp = 0

        self.setup()

    def least_squares_fit(self, x_in, y_out, poly_limits, tolerance):
        """
        Fit a polynomial to a set of x and y data

        Any coefficients already defined are used to seed the new fit.

        - x_in: array of x vectors, one row per vector
        - y_out: vector of y values, should be same length as x_in (i.e. one y value for each x vector)
        - polynomial_order: maximum order of the polynomial, for fitting
        - poly_limits: iterable of length 2, giving the minimum and maximum value of polynomial coefficients
        - tolerance: tolerance on the RMS residual of the coefficients for the optimiser
        Returns polynomial coefficients
        """
        x_in = numpy.array(x_in)
        self.y_out_tmp = numpy.array(y_out)
        if self.dimension != x_in.shape[1]:
            raise ValueError("Input vector lengths did not match dimension")
        self.xp_tmp = self.make_polynomial_vector(x_in)
        self.xp_tmp = numpy.array(self.xp_tmp) # x polynomial
        self.iter_tmp = 0
        n_coefficients = self.xp_tmp.shape[1]
        limits = [(poly_limits[0], poly_limits[1]) for i in range(n_coefficients)]
        seed = self.make_seed(n_coefficients)
        self.print_time = -1
        optimisation = scipy.optimize.differential_evolution(self.fit_function, limits, tol=tolerance, x0=seed, maxiter=self.max_iter)
        self.polynomial_coefficients = optimisation.x
        if self.verbose > 0:
            self.print_time = -1
            self.fit_function(self.polynomial_coefficients)
            print()
        return optimisation

    def function(self, x_in):
        """
        Calculate a vector of y values based on an array of x values
        """
        xpoly = self.make_polynomial_vector(x_in)
        y_out = numpy.dot(xpoly, self.polynomial_coefficients)
        return y_out

    def setup(self):
        pass

    def remove_duplicates(self, indices):
       return numpy.unique(indices, axis=0).tolist()

    def make_polynomial_indices(self):
        """
        Make
        """
        all_indices = [[]]
        this_order_indices = [[]]
        for order in range(self.polynomial_order):
            next_order_indices = []
            for index in this_order_indices:
                for axis in range(self.dimension):
                    new_index = sorted(index+[axis])
                    next_order_indices.append(new_index)
            next_order_indices = self.remove_duplicates(next_order_indices)
            all_indices += next_order_indices
            this_order_indices = next_order_indices
        return all_indices

    def make_polynomial_vector(self, x_in):
        """
        Make a vector of the variables like 1, x, xp, ...., x^2, x^2p, xp^2, ..., x^3, ...
        - x_in: array of variables in the polynomial e.g. [[x, xp, y, yp]]
        Returns the vector as a numpy array.
        """
        x_in = numpy.array(x_in)
        polynomial_indices = self.make_polynomial_indices()
        polynomial_vector = [[
            numpy.prod([x_vector[j] for j in indices]) # x_j0*x_j1*...
                for indices in polynomial_indices] # [0, x_0, x_1, ..., x_00, ...]
                    for x_vector in x_in] # for each x point in x_in
        polynomial_vector = numpy.array(polynomial_vector)
        return polynomial_vector

    def make_seed(self, n_coefficients):
        coefficients = list(self.polynomial_coefficients) + [0.0]*n_coefficients
        coefficients = coefficients[:n_coefficients]
        return coefficients


    def fit_function(self, polynomial_coefficients):
        """
        Calculate the RMS residual of y vs polynomial(x) for use in DE
        """
        coeffs_np = numpy.array(polynomial_coefficients)
        y_out_calc = numpy.dot(self.xp_tmp, coeffs_np)
        residual = self.y_out_tmp - y_out_calc
        residual = residual*residual
        score = numpy.sum(residual)**0.5
        self.iter_tmp += 1
        if self.verbose > 0 and time.time() > self.print_time:
            self.print_time = time.time()+self.print_time_step
            print(f"i: {self.iter_tmp} coeffs: {coeffs_np} score: {score:10.6g}")
        return score
