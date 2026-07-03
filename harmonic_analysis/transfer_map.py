import copy
import math
import numpy
import scipy.special 

import polynomial_fit

"""
Analytical form of the transfer map, *including longitudinal field components*
"""
# methods for polynomials; I should make a polynomial class
def index_equality(poly_index_1, poly_index_2):
    """
    Return true if poly_index_1 refers to the same component as poly_index_2
    """
    return sorted(poly_index_1) == sorted(poly_index_2)


class TransferMap:
    """Should mimic the interface of polynomial_fit.MultipolynomialFit"""
    def __init__(self):
        self.max_order = 3 # octupole
        self.dimension = 4
        self.step_size = 1e-4 # I *think* the units are [m]
        self.momentum = ((70**2+938.272**2)-938.272**2)**0.5 # MeV/c
        self.c_light = 299792458 # m/s
        self.indices = None
        self.x_index = 0
        self.y_index = 2

    def get_indices(self):
        """
        Get the indices for columns in the transfer map
        """
        # borrow the indexing routines from polynomial fit
        pfit = polynomial_fit.PolynomialFit()
        pfit.polynomial_order = self.max_order
        pfit.dimension = self.dimension
        self.indices = pfit.make_polynomial_indices()
        return self.indices

    def calculate_delta_map(self, multipoles, longitudinals):
        """
        Using M = I + dM ds; this returns dM as a numpy array. This only works for quad
        components and rotated quad components (probably not even skew quads)

        where
         * M is the (maybe non-linear) transfer map for a step from s to s + ds
         * ds is the step size
         * I is the identity matrix

        - multipoles is the normalised FFT expansion i.e. the multipole expansion,
          an iterable with elements C_n
        - longitudinals is the longitudinal field components
        """
        dtm = [[0.0 for i in range(len(self.indices))] for j in range(self.dimension)]
        dtm = numpy.array(dtm)
        quad = numpy.real(multipoles[2]) # normal quadrupole
        longi = numpy.real(longitudinals[1]) # rotated normal quadrupole term
        b_units = self.c_light*1e-9 # 1e6 is conversion to MeV/c; not sure what is the other 1e3 - mm??
        # x_out = x_in + x_in' ds
        dtm[0, 2] = 1.0/self.momentum
        # x'_out = x'_in + quad x_in ds + rot_quad  y y'
        dtm[1, 1] = quad * b_units
        dtm[1, 13] = -longi * b_units/self.momentum
        # y_out = y_in + y_in' ds
        dtm[2, 4] = 1.0/self.momentum
        # y'_out = y'_in + quad y_in ds + rot_quad  y x'
        dtm[3, 3] = -quad * b_units
        dtm[3, 10] = longi * b_units/self.momentum
        return dtm

    def get_multipole_polynomial(self, n, cn):
        """
        Return polynomial coefficients for a given multipole
        - n: indexes the multipole; 0 means dipole, 1 means quadrupole, etc
        - cn: n^{th} multipole strength, with real component giving normal
              multipole and imaginary component giving the skew multipole
        Returns a numpy array of length self.indices corresponding to the
        complex polynomial for a multipole. Real component is By, complex
        component is Bx
        """
        multipole_coefficients = numpy.array([0.0j for index in self.indices])
        for a in range(0, n+1):
            b = n-a
            binomial_coeff = scipy.special.binom(n, b)
            the_index = [self.x_index]*a+[self.y_index]*b
            coeff = binomial_coeff*(1j**b)*cn
            for i, an_index in enumerate(self.indices): # inefficient
                if index_equality(an_index, the_index):
                    multipole_coefficients[i] = coeff
        return multipole_coefficients

    def get_multipole_field(self, multipoles):
        """
        Get Bx & By polynomials given a set of multipole strengths
        - multipoles: iterable corresponding to the multipole strengths
        Returns tuple of (bx, by) where bx is an array of polynomial
        coefficients for Bx and by is an array of polynomial coefficients for By.
        bx and by should be real.
        """
        multipole_coefficients = self.get_multipole_polynomial(0, multipoles[0])
        for n, cn in enumerate(multipoles[1:]):
            multipole_coefficients += self.get_multipole_polynomial(n+1, cn)
        bx = numpy.imag(multipole_coefficients)
        by = numpy.real(multipole_coefficients)
        return bx, by

    def rotate_multipole_field(self, bx_coeff, by_coeff, angle):
        """
        Rotate a multipole to get bx, by and bs polynomial
        - bx_coeff: iterable of polynomial coefficients for bx
        - by_coeff: iterable of polynomial coefficients for by
        - angle: rotation angle [rad]
        Returns a tuple of bx, by and bs coefficients in the rotated system
        """
        ctheta = math.cos(angle)
        stheta = math.sin(angle)
        # in the rotated system, x is actually to the reference trajectory
        # e.g. x^2 y becomes [x cos(theta)]^2 y
        for i, index in enumerate(self.indices):
            # index is list like [0, 0, 2] which means x^2 y
            for axis in index:
                if axis == 0:
                    bx_coeff[i] *= ctheta
                    by_coeff[i] *= ctheta
        # in the rotated system, some of bx turns into bs
        bs_coeff = copy.deepcopy(bx_coeff)*stheta
        bx_coeff *= ctheta
        # and return
        return bx_coeff, by_coeff, bs_coeff

    def calculate_delta_map_rotated(self, multipoles, angle):
        """
        Using M = I + dM ds; this returns dM as a numpy array

        where
         * M is the (maybe non-linear) transfer map for a step from s to s + ds
         * ds is the step size
         * I is the identity matrix

        - multipoles is the normalised FFT expansion i.e. the multipole expansion,
          an iterable with elements C_n
        - longitudinals is the longitudinal field components
        """
        self.indices = self.get_indices()

        dtm = [[0.0 for i in range(len(self.indices))] for j in range(self.dimension)]
        dtm = numpy.array(dtm)

        # field in terms of polynomial coefficients
        bx, by = self.get_multipole_field(multipoles)
        bx, by, bs = self.rotate_multipole_field(bx, by, angle)
        b_units = self.c_light*1e-9 # 1e6 is conversion to MeV/c; not sure what is the other 1e3 - mm??
        # x_out = x_in + x_in' ds
        dtm[0, 2] = 1.0/self.momentum
        # x'_out = x'_in + bs(x, y) py ds - by(x, y) ps ds
        dtm[1] = -numpy.array(by) * b_units # -by ps
        for i, index in enumerate(self.indices): # +bs py
            for j, test_index in enumerate(self.indices):
                if index_equality(test_index+[3], index):
                    dtm[1, i] += +bs[j]*b_units/self.momentum

        # y_out = y_in + y_in' ds
        dtm[2, 4] = 1.0/self.momentum
        # y'_out = y'_in + bx ps ds - px bs ds
        dtm[3] = bx * b_units # +bx ps
        for i, index in enumerate(self.indices): # -bs px
            for j, test_index in enumerate(self.indices):
                if index_equality(test_index+[1], index):
                    dtm[3, i] += -bs[j]*b_units/self.momentum

        return dtm

    def calculate_tm(self, multipoles, longitudinals):
        """
        Returns the transfer map from s to s + ds through quadrupole and
        longitudinal field only

        Does not handle non-quadrupole fields correctly, but can take FFT in
        Br, Bs as input. Okay assuming a field map that only has quadrupole.
        Does not work even with a dipole because I haven't put in a coordinate
        system rotation.
        """
        dtm = self.calculate_delta_map(multipoles, longitudinals)*self.step_size
        for i in range(self.dimension):
            dtm[i, i+1] = 1
        return dtm

    def calculate_tm_rotated(self, multipoles, angle):
        """
        Returns the transfer map from s to s + ds through arbitrary multipole
        field including rotation through angle

        handles all multipoles, but user has to enter the non-rotated multipole
        FFT and a rotation angle. Good for testing, but not much use when
        looking at a generalised field map.
        """
        dtm = self.calculate_delta_map_rotated(multipoles, angle)*self.step_size
        for i in range(self.dimension):
            dtm[i, i+1] = 1
        return dtm

