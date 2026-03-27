import os
import shutil

import numpy
import h5py
import scipy
import itertools

import pyopal.objects.minimal_runner
import pyopal.elements.multipolet
import pyopal.elements.output_plane

class Simulation(pyopal.objects.minimal_runner.MinimalRunner):
    def __init__(self):
        """Initialise"""
        super().__init__()

        self.run_name = "tracking_analysis"
        self.tmp_dir = "./tracking_analysis"
        self.verbose = 0

        self.ke = 0.07 # initial kinetic energy [GeV]
        self.max_steps = 5
        self.steps_per_turn = 1000
        self.step_size_m = 1e-3 # tracking step size [metre]

        # particle grid
        self.delta_vector = numpy.array([1e-3, 1e-3, 1e-3, 1e-3]) # size of grid in x, xp, y, yp
        self.particle_grid_order = 3 # 0 is reference only, 1 is linear, 2 is quadratic, etc
        self.negative_grid = True # if False, track particles in positive delta region; if True also track in negative delta region

        # least squares solve

        self.setup()

    def setup(self):
        """Set up derived quantities"""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        self.momentum = ((self.ke+self.mass)**2-self.mass**2)**0.5 # [GeV/c]
        self.time_per_turn = self.steps_per_turn*self.step_size_s(self.step_size_m) # seconds
        self.build_distribution_grid()

    def step_size_s(self, step_size_mm):
        """Step size in seconds"""
        beta_rel = self.momentum/(self.ke+self.mass)
        c_light = 299792458 # m/s
        step_size_ns = step_size_mm/beta_rel/c_light
        return step_size_ns

    def make_element_iterable(self):
        """
        This is just a dummy lattice, so just check that we can add some dummy drifts
        """
        return [self.null_drift(), self.make_output_plane()]

    def make_output_plane(self):
        output_plane = pyopal.elements.output_plane.OutputPlane()
        output_plane.set_attributes(
            centre=[self.r0, 1e-3, 0.0],
            normal=[0.0, 1.0, 0.0],
            placement_style="CENTRE_NORMAL",
            algorithm="RK4",
            tolerance=1e-6,
            height=1,
            output_filename="output",
            verbose_level=0,
            width=1,
        )
        return output_plane

    def generate_particle_meshgrids(self):
        """
        Generate into a meshgrid where each element is a tensor of x, tensor of xp, ...
        """
        order = self.particle_grid_order
        # generate the list of deltas [0, 1*dx, 2*dx, ...] [0, 1*dxp, 2*dxp, ...], ...
        dimension = len(self.delta_vector)
        if self.negative_grid:
            one_d_arrays = numpy.linspace(-order*self.delta_vector, order*self.delta_vector, 2*order+1)
        else:
            one_d_arrays = numpy.linspace(0, order*self.delta_vector, order+1)
        particle_grids = numpy.meshgrid(*(one_d_arrays.transpose()))
        return particle_grids

    def flatten_meshgrids(self, particle_grids):
        particle_vectors = []
        an_iterator = numpy.nditer(particle_grids[0], order='F', flags=['multi_index'])
        for x in an_iterator:
            xp = particle_grids[1][an_iterator.multi_index]
            y = particle_grids[2][an_iterator.multi_index]
            yp = particle_grids[3][an_iterator.multi_index]
            vector = numpy.array([x, xp, y, yp])
            index = numpy.abs(vector/self.delta_vector)
            if sum(index)-self.particle_grid_order > 0.1:
                continue
            particle_vectors.append([x, xp, y, yp])
        particle_vectors = sorted(particle_vectors, key=lambda vec: numpy.abs(vec).tolist())
        return particle_vectors

    def generate_distribution_string(self, particle_vectors):
        self.distribution_str = f"{len(particle_vectors)}\n"
        for a_vector in particle_vectors:
            a_vector = a_vector[0:2]+[0, 0]+a_vector[2:]
            for u in a_vector:
                self.distribution_str += f"{u:12.8g} "
            self.distribution_str += "\n"

    def build_distribution_grid(self):
        """
        """
        phase_space_vector = []
        particle_grids = self.generate_particle_meshgrids()
        particle_vectors = self.flatten_meshgrids(particle_grids)
        self.generate_distribution_string(particle_vectors)

class TrackingAnalysis:
    def __init__(self):
        self.verbose = 100
        self.tmp_dir = "./tracking_analysis"
        self.run_name = "tracking_analysis"
        self.h5_filename = os.path.join(self.tmp_dir, "output.h5")
        self.h5_key_list = ["x", "y", "z", "time", "px", "py", "pz", "id"]
        self.hits = None

        # ephemeral data used during fitting
        self.polynomial_data_tmp = None
        self.coefficients_one_d_tmp = None
        self.dimension_tmp = None

        self.setup()

    def setup(self):
        pass

    def load_beam_file(self):
        self.hits = [hit for hit in self.generate_hit_h5py()]

    def generate_hit_h5py(self):
        h5_file = h5py.File(self.h5_filename, 'r')
        hits = [] # list of hits in the file
        for key in h5_file.keys():
            if key[:5] != "Step#":
                if self.verbose > 10:
                    print("Skipping", key)
                continue
            n_steps = len(h5_file[key]["x"])
            for i in range(n_steps):
                hit_dict = {}
                for h5_key in self.h5_key_list:
                    hit_dict[h5_key] = h5_file[key][h5_key][i]
                if self.verbose >= 4:
                    print(hit_dict)
                yield(hit_dict)

    def remove_duplicates(self, indices):
       return numpy.unique(indices, axis=0).tolist()

    def make_polynomial_indices(self, dimension, polynomial_order):
        all_indices = [[]]
        this_order_indices = [[]]
        for order in range(polynomial_order):
            next_order_indices = []
            for index in this_order_indices:
                for axis in range(dimension):
                    new_index = sorted(index+[axis])
                    next_order_indices.append(new_index)
            next_order_indices = self.remove_duplicates(next_order_indices)
            all_indices += next_order_indices
            this_order_indices = next_order_indices
        return all_indices

    def make_polynomial_vector(self, x_in, polynomial_order):
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
        polynomial_indices = self.make_polynomial_indices(x_in.shape[1], polynomial_order)
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
        self.xp_tmp = self.make_polynomial_vector(x_in, polynomial_order)
        self.xp_tmp = numpy.array(self.xp_tmp) # x polynomial
        self.y_out_tmp = y_out
        n_coefficients = self.xp_tmp.shape[1]
        limits = [(poly_limits[0], poly_limits[1]) for i in range(n_coefficients)]
        optimisation = scipy.optimize.differential_evolution(self.function, limits, tol=tolerance) #, x0=[0.0]*len(limits))
        return optimisation

    def do_plots(self):
        pass

    def function(self, polynomial_coefficients):
        coeffs_np = numpy.array(polynomial_coefficients)
        y_out_calc = numpy.dot(self.xp_tmp, coeffs_np)
        residual = self.y_out_tmp - y_out_calc
        residual = residual*residual
        score = numpy.sum(residual)**0.5
        return score


def main():
    #simulation = Simulation()
    #simulation.execute_fork()
    #print(f"Simulation running in {simulation.tmp_dir}")

    analysis = Analysis()
    analysis.do_plots()

if __name__ == "__main__":
    main()