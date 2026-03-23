import os

import pandas
import numpy
import matplotlib
import matplotlib.pyplot

import pyopal.objects.parser
import pyopal.objects.field

"""
HarmonicAnalysis builds a multipole expansion around a trajectory based on an OPAL input file

Algorithm based on thesis by Max Topp-Mugglestone

Scaling Fixed Field Accelerators: Theory and Modelling of Horizontal- and
Vertical-Excursion Accelerators, Univ. Oxford, 2024

The algorithm is:
* pyopal opens a lattice file and runs the file, including any tracking steps.
* pyopal opens a beam file and loads the beam file
* HarmonicAnalysis loops over the trajectories in the beam file and looks for user-specified trajectory
* For each step in the trajectory
** Calculate the coordinate system
** Generate set of points orthogonal to the trajectory
** Calculate fields on each point, in trajectory coordinate system
** Do FFT on the calculated fields
** Figure out multipole components based on FFT
* Store the list of multipole components

For a field like
        Bz + iBx = C_n (x - i z)^{n-1}
the values of C_n at each step are stored in self.fft_list
"""

class HarmonicAnalysis:
    def __init__(self, config):
        self.config = config
        self.trajectory = None
        self.trajectory_columns = ["id", "x", "xp", "y", "yp", "z", "zp"]
        self.field = pyopal.objects.field # by default we use pyopal field
        self.fft_list = []

    def parse_lattice_file(self):
        """
        Load an OPAL lattice file and execute it to build the field map in memory
        """
        lattice_filename = self.config["lattice_filename"]
        pyopal.objects.parser.initialise_from_opal_file(lattice_filename)

    def parse_track_orbit_file(self):
        """
        Load a trackOrbit file to find the trajectory
        """
        trajectory_filename = self.config["trajectory_filename"]
        columns = self.trajectory_columns
        self.trajectory = pandas.read_csv(trajectory_filename, sep=" ",
                                          header=0, names=columns, skiprows=2)
        self.trajectory["step"] = [i for i in range(self.trajectory.shape[0])]
        print(f"Loaded {self.trajectory.shape[0]} rows")

    def analyse_trajectory(self):
        """
        Loop over the trajectory and perform harmonic analysis for each step
        """
        self.fft_list = []
        for i, step in self.trajectory.iterrows():
            if step["id"] != self.config["harmonic_analysis_track_id"]:
                continue
            will_plot = i in self.config["do_one_step_plot"]
            fft_r, fft_phi, fft_l = self.analyse_one_step(step, will_plot)
            self.fft_list.append(an_fft)

    def analyse_one_step(self, psv, will_plot):
        """
        Calculate the multipole components for one step
        - psv: phase space vector; one line from input trackOrbit file
        - will_plot: set to True to make plots characterising the fft
        Returns the FFT for the step
        """
        horizontal, vertical, longitudinal = self.get_coordinate_system(psv)
        centre, coordinates = self.build_rotated_vectors(psv, horizontal, vertical)
        b_cyl = self.calculate_field_cyl(longitudinal, centre, coordinates)
        fft_cyl = self.calculate_fft_cyl(b_cyl)
        if will_plot:
            b_cart = self.calculate_field_cart(horizontal, vertical, longitudinal, coordinates)
            self.plot_one_step(psv, horizontal, vertical, coordinates, b_cart, b_cyl, fft_cyl)
        return fft_cyl

    def get_p_vector(self, psv):
        """
        Get the normalised momentum vector for input row psv
        - psv: phase space vector; one line from input trackOrbit file
        Returns momentum vector normalised to length one
        """
        # position and direction vectors
        p_vector = self.norm(numpy.array([psv["xp"], psv["yp"], psv["zp"]]))
        return p_vector

    def get_x_vector(self, psv):
        """
        Get the position vector for input row psv
        - psv: phase space vector; one line from input trackOrbit file
        """
        x_vector = numpy.array([psv["x"], psv["y"], psv["z"]])
        return x_vector

    def get_coordinate_system(self, psv):
        """
        Get the orthogonal coordinate system for a given phase space vector
        - psv: phase space vector; one line from input trackOrbit file
        Returns horizontal, vertical and longitudinal unit vectors
        """
        down = [0,0,-1]
        p_vector = self.get_p_vector(psv)
        # we define horizontal as the vector perpendicular to p_vector and down
        # note that if p_vector == +-down, this algorithm will fail
        horizontal = self.norm(numpy.cross(down, p_vector))
        # we define vertical as the vector perpendicular to p_vector and horizontal
        vertical = self.norm(numpy.cross(horizontal, p_vector))
        return horizontal, vertical, p_vector

    def get_angles(self):
        """
        Get an array of angles, of length self.config["harmonic"] from 0 <= t < 2 pi
        """
        angles = numpy.linspace(0, 2*numpy.pi, self.config["harmonic"], endpoint=False)
        return angles

    def build_rotated_vectors(self, psv, horizontal, vertical):
        """
        Set up the vectors at which we calculate the field values
        - psv: phase space vector; one line from input trackOrbit file
        - horizontal: unit vector in horizontal direction
        - vertical: unit vector in vertical direction
        Returns a list of coordinates in a circle centred on psv position in the
        plane defined by horizontal and vertical vectors
        """
        centre = self.get_x_vector(psv)
        delta = self.config["delta"]
        # angles is the list from [0, dtheta, 2dtheta, ..., 2 pi - dtheta]
        angles = self.get_angles()
        # I *think* we don't want the 2 pi in the list otherwise we double count this point
        # we define the offset vectors as cos(theta)*horizontal + sin(theta)*vertical
        angles = numpy.array([angles]).transpose()
        coordinates = numpy.cos(angles)*horizontal+numpy.sin(angles)*vertical
        coordinates *= delta
        coordinates += centre
        # scale coordinates to delta
        return centre, coordinates

    def calculate_field_global(self, coordinates):
        """
        Calculate the field on coordinates in the global coordinate system
        - coordinates: list of coordinates at which the field will be calculated
        Returns an array like [[bx], [by], [bz]]
        """
        bfield = None
        # loop over the points and calculate bfield in global coordinates
        for point in coordinates:
            # x, y, z, t
            point_tuple = point[0], point[1], point[2], 0.0
            # get_field_value returns out_of_bounds, B 3-vector, E 3-vector
            field = self.field.get_field_value(*point_tuple)
            a_field = numpy.array([field[1:4]])
            if bfield is None:
                bfield = a_field
            else:
                bfield = numpy.concatenate([bfield, a_field])
        bfield = bfield.transpose()
        return bfield


    def calculate_field_cart(self, horizontal, vertical, longitudinal, coordinates):
        """
        Calculate the field on coordinates in the cartesian coordinate system
        defined by horizontal, vertical, longitudinal
        - horizontal: unit vector in horizontal direction
        - vertical: unit vector in vertical direction
        - longitudinal: unit vector in longitudinal direction
        - coordinates: list of coordinates at which field will be calculated
        Returns [[bh], [bv], [bl]]; fields in horizontal, vertical and longitudinal
        directions respectively, as a numpy array.
        """
        bfield = self.calculate_field_global(coordinates)
        bfield = bfield.transpose()
        # transform to field in coordinate system orthogonal to the psv
        bh = numpy.dot(bfield, horizontal)
        bv = numpy.dot(bfield, vertical)
        bl = numpy.dot(bfield, longitudinal)
        return numpy.array([bh, bv, bl])

    def calculate_field_cyl(self, longitudinal, centre, coordinates):
        """
        Calculate the field as cylindrical components (r, phi, l)
        - longitudinal: the longitudinal unit vector
        - centre: centre of the coordinate system (e.g. position of the particle)
        - coordinates: list of coordinates on which field will be calculated
        Returns array of [[br], [bphi], [bl]]; field component in cylindrical
        coordinates
        """
        br, bphi, bl = [], [], []
        bfield = self.calculate_field_global(coordinates)
        bfield = bfield.transpose()
        # loop over bfield and do the appropriate dot products
        # there may be a quicker way to do this using clever matrix stuff but
        # this probably works
        for i, a_field in enumerate(bfield):
            point = coordinates[i]
            r_v = self.norm(point-centre)
            # warning - CHECK phi sign here
            phi_v = self.norm(numpy.cross(longitudinal, point-centre))
            br.append(numpy.dot(a_field, r_v))
            bphi.append(numpy.dot(a_field, phi_v))
            bl.append(numpy.dot(a_field, longitudinal))
        b_cyl = numpy.array([br, bphi, bl])
        return b_cyl

    def fft_normalisation(self, i):
        """
        Return normalisation factor for i^th harmonic
        - i: index of harmonic

        Note that for small values of delta (e.g. 1e-3) and large values of
        harmonic (e.g. 256) we get floating point precision type errors.
        """
        dr = self.config["delta"]
        harm = self.config["harmonic"]
        pow_index = min(i, harm-i)
        dr_pow = dr**(pow_index-1) # maximum value here is like config["delta"]**(config["harmonic"]/2)
        if dr_pow == 0.0 or dr_pow != dr_pow:
            raise ValueError("Floating point precision error when running "
                f"harmonic {harm}, delta {dr} giving normalisation {dr_pow}")
        normalisation = 2j/harm/dr_pow
        return normalisation


    def calculate_fft_cyl(self, b_cyl):
        """
        Do the FFT
        - b_cyl: array of field values. Rows correspond to field values for a
        direction; columns for a point. So expect 3 rows each with [harmonic]
        elements
        Returns fft of each row, normalised by r^(i-1)/n. Shape is same as b_cyl
        """
        fft_cyl = numpy.fft.fft(b_cyl)
        norm = numpy.array([self.fft_normalisation(i) for i in range(self.config["harmonic"])])
        fft_cyl = fft_cyl*norm
        return fft_cyl

    def plot_one_step(self, psv, horizontal, vertical, coordinates, b_cart, b_cyl, fft_cyl):
        """
        Make plots characterising harmonic analysis of a single step
        """
        print("Plotting at", psv)
        figure = self.plot_coordinate_system(psv, horizontal, vertical, coordinates)
        figure.suptitle(f"Step {psv['step']}")
        figure.savefig(f"coords_{psv['step']}.png")
        figure = self.plot_fields(b_cart, b_cyl)
        figure.suptitle(f"Step {psv['step']}")
        figure.savefig(f"fields_{psv['step']}.png")
        for i, axis in [(0, "r"), (1, "$\\phi$"), (2, "longitudinal"), ]:
            name = axis.replace("$", "").replace("\\", "")
            figure = self.plot_fft(fft_cyl[i], axis)
            figure.suptitle(f"Step {psv['step']}")
            figure.savefig(f"fft_{psv['step']}_{name}.png")

    def plot_coordinate_system(self, psv, horizontal, vertical, coordinates):
        """
        Plot the coordinate system
        - psv: phase space vector; one line from input trackOrbit file
        - horizontal: unit vector in horizontal direction, perpendicular to p
        - vertical: unit vector in vertical direction, perpendicular to p
        - coordinates: coordinates on which the field is calculated
        Returns the figure
        """
        delta = self.config["delta"]
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(projection='3d')
        for item in coordinates:
            plot = coordinates.transpose()
            axes.scatter(plot[0], plot[1], plot[2])
        for vector, name in [
            (self.get_p_vector(psv), "longitudinal"),
            (horizontal, "horizontal"),
            (vertical, "vertical")]:
            x0 = self.get_x_vector(psv)
            x1 = vector*delta+x0
            coordinates = numpy.array([x0, x1]).transpose()
            axes.plot(coordinates[0], coordinates[1], coordinates[2], label=name)
        axes.legend()

        axes.set_xlim([psv["x"]-delta*1.5, psv["x"]+delta*1.5])
        axes.set_ylim([psv["y"]-delta*1.5, psv["y"]+delta*1.5])
        axes.set_zlim([psv["z"]-delta*1.5, psv["z"]+delta*1.5])
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_zlabel("z [m]")
        return figure

    def plot_fields(self, b_cart, b_cyl):
        """
        Plot the fields
        - bh: list of horizontal field values
        - bv: list of vertical field values
        - bl: list of longitudinal field values
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        bh, bv, bl1 = b_cart
        br, bphi, bl2 = b_cyl
        for bfield, name, style in [(bh, "$B_h$", "--"), (bv, "$B_v$", "--"), (bl1, "$B_{lx}$", "--"),
                (br, "$B_r$", ":"), (bphi, "$B_{\\phi}$", ":"), (bl2, "$B_{lr}$", ":")]:
            angles = numpy.degrees(self.get_angles())
            axes.plot(angles, bfield, label=name, linestyle=style)
        ylim = axes.get_ylim()
        for angle in [0.0, 90.0, 180, 270.0, 360.0]:
            axes.plot([angle, angle], ylim, linestyle="-", color="xkcd:light grey")
        axes.set_ylim(ylim)
        axes.set_xlabel("angle [degree]")
        axes.set_ylabel("B [T]")
        axes.legend()
        return figure

    def plot_fft(self, a_fft, fft_axis):
        """
        Plot the fourier transform
        - a_fft: fourier transform output in one dimension (i.e. a vector of complex numbers)
        - fft_axis: String for the title (e.g. r, phi, longitudinal)
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        x_axis = [i for i, z in enumerate(a_fft)]
        axes.scatter(x_axis, numpy.real(a_fft), label="real")
        axes.scatter(x_axis, numpy.imag(a_fft), label="imaginary")
        axes.legend()
        axes.set_title(f"Axis: {fft_axis}")
        return figure

    @classmethod
    def norm(self, a_vector):
        """Normalise a vector"""
        return a_vector/numpy.linalg.norm(a_vector)

def default_config():
    config = {
        "working_directory":"lattice/",
        "lattice_filename":"benchmark_lattice-20260216_placement-testing",
        "trajectory_filename":"benchmark_lattice-20260216_placement-testing-trackOrbit.dat",
        "harmonic_analysis_track_id":"ID1", # ID of trajectory
        "harmonic":8, # number of terms in FFT
        "delta":1e-3, # distance from trajectory to harmonic analysis coordinates [metres]
        "do_one_step_plot":[100],
    }
    return config

def main():
    config = default_config()
    here = os.getcwd()
    os.chdir(config["working_directory"])
    analysis = HarmonicAnalysis(config)
    analysis.parse_lattice_file()
    analysis.parse_track_orbit_file()
    analysis.analyse_trajectory()
    os.chdir(here)

if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <Enter> to finish")

