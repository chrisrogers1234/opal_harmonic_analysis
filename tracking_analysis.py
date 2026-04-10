import os
import shutil

import numpy
import h5py
import scipy
import itertools

import pyopal.objects.minimal_runner
import pyopal.elements.multipolet
import pyopal.elements.output_plane

import polynomial_fit

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
        self.particle_algorithm = "grid" # "grid" or "scatter"

        # particle grid
        self.delta_vector = numpy.array([1e-1, 1e-1, 1e-1, 1e-1]) # size of grid in x, xp, y, yp
        self.particle_grid_order = 3 # 0 is reference only, 1 is linear, 2 is quadratic, etc
        self.negative_grid = True # if False, track particles in positive delta region; if True also track in negative delta region

        # particle scatter
        self.n_particles = 1000

    def setup(self):
        """Set up derived quantities"""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        self.momentum = ((self.ke+self.mass)**2-self.mass**2)**0.5 # [GeV/c]
        self.time_per_turn = self.steps_per_turn*self.step_size_s(self.step_size_m) # seconds
        if self.particle_algorithm == "grid":
            if self.verbose > 3:
                print("Building grid")
            self.build_distribution_grid()
        else:
            if self.verbose > 3:
                print("Building scatter")
            self.build_distribution_scatter()

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
        return [self.null_drift(), self.make_output_plane(1e-9, "plane_0"), self.make_output_plane(1e-3, "plane_1")]

    def make_output_plane(self, y0, filename):
        output_plane = pyopal.elements.output_plane.OutputPlane()
        output_plane.set_attributes(
            centre=[self.r0, y0, 0.0],
            normal=[0.0, 1.0, 0.0],
            placement_style="CENTRE_NORMAL",
            algorithm="RK4",
            tolerance=1e-15,
            height=1,
            output_filename=filename,
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
            particle_vectors.append(vector)
        particle_vectors = sorted(particle_vectors, key=lambda vec: numpy.abs(vec).tolist())
        return particle_vectors

    def generate_distribution_string(self, particle_vectors):
        self.distribution_str = f"{len(particle_vectors)}\n"
        for a_vector in particle_vectors:
            a_vector = list(a_vector)
            a_vector = a_vector[0:2]+[0, 0]+a_vector[2:]
            for u in a_vector:
                self.distribution_str += f"{u:12.8g} "
            self.distribution_str += "\n"

    def build_distribution_grid(self):
        """
        """
        particle_grids = self.generate_particle_meshgrids()
        particle_vectors = self.flatten_meshgrids(particle_grids)
        self.generate_distribution_string(particle_vectors)

    def build_distribution_scatter(self):
        particle_vectors = numpy.array([numpy.random.uniform(-d, d, self.n_particles) for d in self.delta_vector])
        particle_vectors = particle_vectors.transpose()
        self.generate_distribution_string(particle_vectors)


class TrackingAnalysis:
    def __init__(self):
        self.verbose = 100
        self.tmp_dir = "./tracking_analysis"
        self.run_name = "tracking_analysis"
        self.h5_filename_in = os.path.join(self.tmp_dir, "plane_0.h5")
        self.h5_filename_out = os.path.join(self.tmp_dir, "plane_1.h5")
        self.h5_key_list = ["x", "y", "z", "time", "px", "py", "pz", "id"]
        self.hits_in = None
        self.hits_out = None
        self.fit_order = 2
        self.limits = [-10, 10]
        self.chi2_tolerance = 1e-15
        self.polynomial_fit = None
        self.delta_vector = numpy.array([1e-1, 1e-1, 1e-1, 1e-1])


    def load_beam_files(self):
        # load the hits
        self.hits_in = [hit for hit in self.generate_hit_h5py(self.h5_filename_in)]
        self.hits_out = [hit for hit in self.generate_hit_h5py(self.h5_filename_out)]
        # find the event ids common to both hit lists
        valid_ids = set([hit["id"] for hit in self.hits_in]) & set([hit["id"] for hit in self.hits_out])
        # generate the array of x, px, y, py values
        array_in = [[hit["x"]/, hit["px"], hit["z"], hit["pz"]] for hit in self.hits_in if hit["id"] in valid_ids]
        array_out = [[hit["x"], hit["px"], hit["z"], hit["pz"]] for hit in self.hits_out if hit["id"] in valid_ids]
        if self.verbose > 4:
            print("x_in\n", numpy.array(array_in))
            print()
            print("x_out\n", numpy.array(array_out))
        # do the polynomial fit
        self.polynomial = polynomial_fit.MultipolynomialFit()
        self.polynomial.dimension = 4
        for fit_order in range(0, self.fit_order+1):
            self.polynomial.polynomial_order = fit_order
            self.polynomial.least_squares_fit(array_in, array_out, self.limits, self.chi2_tolerance)
        print("polynomial\n", self.polynomial.polynomial_coefficients)
        print("x_calc\n", self.polynomial.function(array_in))
        print("score\n", self.polynomial.fit_function(self.polynomial.polynomial_coefficients))

    def generate_hit_h5py(self, h5_filename):
        h5_file = h5py.File(h5_filename, 'r')
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

    def do_plots(self):
        self.load_beam_files()


def main():
    simulation = Simulation()
    simulation.verbose = 0
    simulation.n_particles = 100
    simulation.particle_algorithm = "scatter"
    #simulation.particle_grid_order = 1
    #simulation.particle_algorithm = "grid"
    simulation.setup()
    simulation.execute_fork()
    print(f"Simulation running in {simulation.tmp_dir}")

    analysis = TrackingAnalysis()
    analysis.verbose = 10
    analysis.fit_order = 1
    analysis.do_plots()

    analysis.fit_order = 2
    analysis.do_plots()


if __name__ == "__main__":
    main()