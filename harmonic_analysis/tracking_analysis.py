import os
import shutil
import itertools
import math

import numpy
import h5py
import scipy
import matplotlib
import matplotlib.pyplot

import pyopal.objects.minimal_runner
import pyopal.elements.multipolet
import pyopal.elements.output_plane
import pyopal.elements.local_cartesian_offset

import polynomial_fit

def clear_dir(a_dir):
    if os.path.exists(a_dir):
        shutil.rmtree(a_dir)
    os.makedirs(a_dir)

class Simulation(pyopal.objects.minimal_runner.MinimalRunner):
    def __init__(self):
        """Initialise"""
        super().__init__()

        self.run_name = "tracking_analysis"
        self.tmp_dir = "./tracking_analysis"
        self.plot_dir = "./tracking_analysis"
        self.verbose = 0
        self.r0 = 2.0

        self.ke = 0.07 # initial kinetic energy [GeV]
        self.steps_per_turn = 1000
        self.step_size_m = 1e-4 # tracking step size [metre]
        self.particle_algorithm = "grid" # "grid" or "scatter"
        self.output_separation = 10*self.step_size_m # distance between output planes
        self.max_steps = int(self.output_separation*1.5/self.step_size_m)

        # particle grid
        self.delta_vector = numpy.array([1e-3, 1e6, 1e-3, 1e6]) # size of grid in x, px, y, py # metres, eV
        self.particle_grid_order = 3 # 0 is reference only, 1 is linear, 2 is quadratic, etc
        self.negative_grid = True # if False, track particles in positive delta region; if True also track in negative delta region

        # particle scatter
        self.n_particles = 100

        # fields
        self.multipoles = []
        self.angle = 0.0 # degree, 0.0 means aligned with s axis, +90.0 means aligned with horizontal axis

    def setup(self):
        """Set up derived quantities"""
        clear_dir(self.tmp_dir)
        clear_dir(self.plot_dir)
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
        if self.verbose > 10:
            print(self.distribution_str)

    def step_size_s(self, step_size_mm):
        """Step size in seconds"""
        beta_rel = self.momentum/(self.ke+self.mass)
        c_light = 299792458 # m/s
        step_size_ns = step_size_mm/beta_rel/c_light
        return step_size_ns

    def make_multipole(self):
        if len(self.multipoles) == 0:
            return []
        half_step = pyopal.elements.local_cartesian_offset.LocalCartesianOffset()
        half_step.set_attributes(
            end_position_y=self.step_size_m/2.0,
            end_normal_y=1.0
        )

        translation = pyopal.elements.local_cartesian_offset.LocalCartesianOffset()
        translation.set_attributes(
            end_position_y=-0.5,
            end_normal_y=1.0
        )

        rotation = pyopal.elements.local_cartesian_offset.LocalCartesianOffset()
        rotation.set_attributes(
            end_normal_x=math.sin(math.radians(self.angle)),
            end_normal_y=math.cos(math.radians(self.angle))
        )

        multipole = pyopal.elements.multipolet.MultipoleT()
        multipole.set_attributes(
            horizontal_aperture=self.r0,
            vertical_aperture=self.r0,
            left_fringe=1e-6,
            right_fringe=1e-6,
            length=1,
            maximum_f_order=1,
            maximum_x_order=1,
            t_p=self.multipoles
        )
        return [rotation, translation, multipole] #half_step,

    def make_element_iterable(self):
        """
        This is just a dummy lattice, so just check that we can add some dummy drifts
        """
        line = [
            self.null_drift(),
            self.make_output_plane(1e-9, "plane_0"),
            self.make_output_plane(self.output_separation, "plane_1")
        ]
        line += self.make_multipole()
        return line


    def make_output_plane(self, y0, filename):
        output_plane = pyopal.elements.output_plane.OutputPlane()
        output_plane.set_attributes(
            centre=[self.r0, y0, 0.0],
            normal=[0.0, 1.0, 0.0],
            placement_style="CENTRE_NORMAL",
            algorithm="RK4",
            tolerance=1e-15,
            height=self.r0,
            output_filename=filename,
            verbose_level=0,
            width=self.r0,
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
        particle_vectors = numpy.concatenate([numpy.array([[0.0,0.0,0.0,0.0]]), particle_vectors])
        self.generate_distribution_string(particle_vectors)

class TrackingAnalysis:
    def __init__(self):
        self.verbose = 100
        self.tmp_dir = "./tracking_analysis"
        self.run_name = "tracking_analysis"
        self.h5_filename_in = os.path.join(self.tmp_dir, "plane_0.h5")
        self.h5_filename_out = os.path.join(self.tmp_dir, "plane_1.h5")
        self.h5_key_list = ["x", "y", "z", "time", "px", "py", "pz", "id"]
        self.var_list = ["x", "px", "z", "pz"]
        self.hits_in = None
        self.hits_out = None
        self.fit_order = 1
        self.limits = [-10, 10]
        self.chi2_tolerance = 1e-15
        self.polynomial_fit = None
        self.algorithm = "differential_evolution"


    def generate_hit_vector(self, hit_list, valid_ids):
        """
        Iterate over hit list:
            check that hit id is in valid ids
            append x, px, z, pz to an array
        """
        hit_list = sorted(hit_list, key = lambda hit: hit["id"])
        if hit_list[0]["id"] != 0:
            raise RuntimeError("Expected hit id 0 for reference trajectory")
        ref = hit_list[0]
        hits_gen = [[hit[var]-ref[var] for var in self.var_list] for hit in hit_list if hit["id"] in valid_ids]
        array_data = numpy.array(hits_gen)
        return array_data

    def load_beam_files(self):
        """
        Load the hits

        Note that automatically works in deviation variables for numerical
        stability reasons i.e. subtract reference (id=0) trajectory from other
        trajectories
        """
        # load the hits
        self.hits_in = self.generate_hit_h5py(self.h5_filename_in)
        self.hits_out = self.generate_hit_h5py(self.h5_filename_out)
        # find the event ids common to both hit lists
        valid_ids = set([hit["id"] for hit in self.hits_in]) & set([hit["id"] for hit in self.hits_out])
        # generate the arrays for polynomial solve
        array_in = self.generate_hit_vector(self.hits_in, valid_ids)
        array_out = self.generate_hit_vector(self.hits_out, valid_ids)
        if array_in.shape != array_out.shape:
            raise ValueError(f"Input array shape {array_in.shape} was not the same as output array shape {array_out.shape}")
        if self.verbose > 4:
            print(f"x_in {array_in.shape}\n", numpy.array(array_in[:10]))
            print()
            print(f"x_out {array_out.shape}\n", numpy.array(array_out[:10]))
        # do the polynomial fit
        self.polynomial = polynomial_fit.MultipolynomialFit()
        self.polynomial.algorithm = self.algorithm
        self.polynomial.dimension = len(self.var_list)
        self.polynomial.verbose = self.verbose
        for fit_order in range(0, self.fit_order+1):
            self.polynomial.polynomial_order = fit_order
        self.polynomial.least_squares_fit(array_in, array_out, self.limits, self.chi2_tolerance)
        if self.verbose > 0:
            print("Found polynomial\n", self.polynomial.get_coefficients())
            print("    In ", [p.iter_tmp for p in self.polynomial.one_d_polynomials], "function calls")
            print("    RMS residual", [o.fun for o in self.polynomial.optimisation_list])
        self.residuals = self.polynomial.function(array_in)-array_out

    def generate_hit_h5py(self, h5_filename):
        h5_file = h5py.File(h5_filename, 'r')
        hits = []
        hit_ids = []
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
                    if h5_key in self.unit_conversion:
                        hit_dict[h5_key] *= self.unit_conversion[h5_key]
                if hit_dict["id"] in hit_ids:
                    continue
                hit_ids.append(hit_dict["id"])
                hits.append(hit_dict)
        hits = sorted(hits, key = lambda x: x["id"])
        for hit_dict in hits:
            if self.verbose > 5:
                if self.verbose >= 10:
                    print(hit_dict)
                elif hit_dict["id"] < 2:
                    print(hit_dict)
        return hits

    def max_res_text(self):
        text = "Max residuals:\n"
        abs_residuals = numpy.abs(self.residuals)
        for i in range(4):
            max_res = max(abs_residuals[:, i])
            text += self.axis_labels[i]+f": {max_res:12.4g}; "
        return text

    def do_plots(self, filename):
        self.load_beam_files()
        figure = matplotlib.pyplot.figure(figsize=[20,10])
        figure.suptitle(self.max_res_text())
        for j in range(4):
            for i in range(4):
                axes = figure.add_subplot(4, 4, i+j*4+1)
                axes.set_position([0.1+0.2*i, 0.1+0.2*j, 0.2, 0.2])
                self.plot_residuals(axes, j, i)
                if j > 0:
                    axes.get_xaxis().set_visible(False)
                if i > 0:
                    axes.get_yaxis().set_visible(False)
        figure.savefig(filename)

    def plot_residuals(self, axes, axis_out, axis_in):
        var_in = self.variables[axis_in]
        residual_view = self.residuals[:, axis_out]
        hit_variables = [hit[var_in] for hit in self.hits_in]
        axes.scatter(hit_variables, residual_view)
        axes.set_xlabel(self.axis_labels[axis_in])
        ordinate = self.get_magnitude(residual_view)
        axes.set_ylabel("Res. "+self.axis_labels[axis_out]+f" $\\times 10^{{{ordinate}}}$")

    def get_magnitude(self, x_list):
        x_max = max([abs(x) for x in x_list])
        i = 0
        while x_max/10**i < 1:
            i -= 1
        while x_max/10**i > 10:
            i += 1
        return i

    # converts h5 files to mm, MeV/c
    p_mass = 938.27208943
    unit_conversion = {"x":1000, "px":p_mass, "y":1000, "py":p_mass, "z":1000, "pz":p_mass}
    variables = ["x", "px", "z", "pz"]
    axis_labels = ["h [mm]", "$p_h$ [MeV/c]", "v [mm]", "$p_v$ [MeV/c]"]

def main():
    algorithms = "Powell", "Nelder-Mead", "CG"

    numpy.set_printoptions(linewidth=200)
    simulation = Simulation()
    simulation.verbose = 0
    simulation.n_particles = 100
    simulation.particle_algorithm = "scatter"
    #simulation.particle_grid_order = 1
    #simulation.particle_algorithm = "grid"
    simulation.multipoles = [0.0, 1.0]
    simulation.setup()
    simulation.execute_fork()
    print(f"Simulation running in {simulation.tmp_dir}")

    analysis = TrackingAnalysis()
    analysis.algorithm = "differential_evolution"
    analysis.fit_order = 1
    analysis.verbose = 5
    analysis.do_plots(f"{self.plot_dir}/residuals_order_1.png")
    analysis.fit_order = 2
    analysis.do_plots(f"{self.plot_dir}/residuals_order_2.png")


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <Enter> to finish")