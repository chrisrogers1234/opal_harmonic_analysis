import os
import shutil

import numpy
import h5py
import scipy
import itertools

import pyopal.objects.minimal_runner
import pyopal.elements.multipolet
import pyopal.elements.output_plane

class PolynomialFit:
    def __init__(self):
        self.polynomial_order = 0
        self.dimension = 4
        self.polynomial_coefficients = numpy.array([1])
        # ephemeral data used during fitting
        self.polynomial_data_tmp = None
        self.coefficients_one_d_tmp = None

        self.setup()

    def setup(self):
        pass

    def remove_duplicates(self, indices):
       return numpy.unique(indices, axis=0).tolist()

    def make_polynomial_indices(self):
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
        - polynomial_order: the order of the polynomial. < 1 returns [1],
                            1 is linear, 2 is quadratic etc.
        Returns the vector as a numpy array.
        """
        # BUG duplicates in the array
        # store a vector like u_i1*u_i2 ...
        # to calculate the next order of the vector, we take each element from
        # the vector and multiply by u_1, then u_2, then ... u_n
        # initial value is [1]
        # second loop is 1*[x, xp, y, yp]
        # third loop is x*[x, xp, y, yp]+xp*[xp, y, yp] + ...
        # and so on
        x_in = numpy.array(x_in)
        polynomial_indices = self.make_polynomial_indices()
        polynomial_vector = [[
            numpy.prod([x_vector[j] for j in indices]) # x_j0*x_j1*...
                for indices in polynomial_indices] # [0, x_0, x_1, ..., x_00, ...]
                    for x_vector in x_in] # for each x point in x_in
        polynomial_vector = numpy.array(polynomial_vector)
        return polynomial_vector

    def least_squares_fit(self, x_in, y_out, polynomial_order, poly_limits, tolerance):
        """
        Fit a polynomial to a set of x and y data
        - x_in: array of x vectors, one row per vector
        - y_out: vector of y values, should be same length as x_in (i.e. one y value for each x vector)
        - polynomial_order: maximum order of the polynomial, for fitting
        - poly_limits: iterable of length 2, giving the minimum and maximum value of polynomial coefficients
        - tolerance: tolerance on the RMS residual of the coefficients for the optimiser
        Returns polynomial coefficients
        """
        self.polynomial_order = polynomial_order
        self.dimension = x_in.shape[1]
        self.xp_tmp = self.make_polynomial_vector(x_in)
        self.xp_tmp = numpy.array(self.xp_tmp) # x polynomial
        self.y_out_tmp = y_out
        n_coefficients = self.xp_tmp.shape[1]
        limits = [(poly_limits[0], poly_limits[1]) for i in range(n_coefficients)]
        optimisation = scipy.optimize.differential_evolution(self.fit_function, limits, tol=tolerance)
        self.polynomial_coefficients = optimisation.x
        return optimisation

    def do_plots(self):
        pass

    def fit_function(self, polynomial_coefficients):
        """
        Calculate the RMS residual of y vs polynomial(x) for use in DE
        """
        coeffs_np = numpy.array(polynomial_coefficients)
        y_out_calc = numpy.dot(self.xp_tmp, coeffs_np)
        residual = self.y_out_tmp - y_out_calc
        residual = residual*residual
        score = numpy.sum(residual)**0.5
        return score

    def function(self, x_in):
        """
        Calculate a vector of y values based on an array of x values
        
        """
        xpoly = self.make_polynomial_vector(x_in)
        y_out = numpy.dot(xpoly, coeffs_np)
        return y_out


def main():
    #simulation = Simulation()
    #simulation.execute_fork()
    #print(f"Simulation running in {simulation.tmp_dir}")

    analysis = Analysis()
    analysis.do_plots()

if __name__ == "__main__":
    main()