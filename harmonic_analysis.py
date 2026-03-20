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
** Not implemented: figure out multipole components based on FFT
* Store a list of multipole components
"""

class HarmonicAnalysis:
    def __init__(self, config):
        self.config = config
        self.trajectory = None
        self.trajectory_columns = ["id", "x", "xp", "y", "yp", "z", "zp"]
        self.field = pyopal.objects.field # by default we use pyopal field
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
        for i, step in self.trajectory.iterrows():
            if step["id"] != self.config["harmonic_analysis_track_id"]:
                continue
            will_plot = i in self.config["do_one_step_plot"]
            self.analyse_one_step(step, will_plot)

    def analyse_one_step(self, psv, will_plot):
        """
        Calculate the multipole components for one step
        - psv: phase space vector; one line from input trackOrbit file
        - will_plot: set to True to make plots characterising the fft
        Returns the FFT for the step
        """
        horizontal, vertical, longitudinal = self.get_coordinate_system(psv)
        coordinates = self.build_rotated_vectors(psv, horizontal, vertical)
        bh, bv, bl = self.calculate_field(horizontal, vertical, longitudinal, coordinates)
        a_fft = self.calculate_fft(bh, bv)
        if will_plot:
            self.plot_one_step(psv, horizontal, vertical, coordinates, bh, bv, bl, a_fft)
        return a_fft

    def get_p_vector(self, psv):
        """
        Get the normalised momentum vector for input row psv
        - psv: phase space vector; one line from input trackOrbit file
        Returns momentum vector normalised to length one
        """
        # position and direction vectors
        p_vector = numpy.array([psv["xp"], psv["yp"], psv["zp"]])
        # normalise p_vector (momentum vector)
        p_vector = p_vector/numpy.linalg.norm(p_vector)
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
        horizontal = numpy.cross(down, p_vector)
        # normalise to length 1
        horizontal = horizontal/numpy.linalg.norm(horizontal)
        # we define vertical as the vector perpendicular to p_vector and horizontal
        vertical = numpy.cross(horizontal, p_vector)
        vertical = vertical/numpy.linalg.norm(vertical)
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
        x_vector = self.get_x_vector(psv)
        delta = self.config["delta"]
        # angles is the list from [0, dtheta, 2dtheta, ..., 2 pi - dtheta]
        angles = self.get_angles()
        # I *think* we don't want the 2 pi in the list otherwise we double count this point
        # we define the offset vectors as cos(theta)*horizontal + sin(theta)*vertical
        angles = numpy.array([angles]).transpose()
        coordinates = numpy.cos(angles)*horizontal+numpy.sin(angles)*vertical
        coordinates *= delta
        coordinates += x_vector
        # scale coordinates to delta
        return coordinates

    def calculate_field(self, horizontal, vertical, longitudinal, coordinates):
        """
        Calculate the fields as horizontal and vertical components
        - horizontal: unit vector in horizontal direction
        - vertical: unit vector in vertical direction
        - longitudinal: unit vector in longitudinal direction
        - coordinates: list of coordinates on which field will be calculated
        Returns bh, bv, bl; fields in horizontal, vertical and longitudinal
        directions respectively.
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
        # transform to field in coordinate system orthogonal to the psv
        bh = numpy.dot(bfield, horizontal)
        bv = numpy.dot(bfield, vertical)
        bl = numpy.dot(bfield, longitudinal)
        return bh, bv, bl

    def calculate_fft(self, bh, bv):
        """
        Do the FFT
        """
        complex_array = bh + 1j*bv
        fft = numpy.fft.fft(complex_array)
        return fft

    def plot_one_step(self, psv, horizontal, vertical, coordinates, bh, bv, bl, a_fft):
        """
        Make plots characterising harmonic analysis of a single step
        """
        print("Plotting at", psv)
        figure = self.plot_coordinate_system(psv, horizontal, vertical, coordinates)
        figure.suptitle(f"Step {psv['step']}")
        figure.savefig(f"coords_{psv['step']}.png")
        figure = self.plot_fields(bh, bv, bl)
        figure.suptitle(f"Step {psv['step']}")
        figure.savefig(f"fields_{psv['step']}.png")
        figure = self.plot_fft(a_fft)
        figure.suptitle(f"Step {psv['step']}")
        figure.savefig(f"fft_{psv['step']}.png")

    def plot_coordinate_system(self, psv, horizontal, vertical, coordinates):
        """
        Plot the coordinate system
        - psv: phase space vector; one line from input trackOrbit file
        - horizontal: unit vector in horizontal direction, perpendicular to p
        - vertical: unit vector in vertical direction, perpendicular to p
        - coordinates: coordinates on which the field is calculated
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

    def plot_fields(self, bh, bv, bl):
        """
        Plot the fields
        - bh: list of horizontal field values
        - bv: list of vertical field values
        - bl: list of longitudinal field values
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        for bfield, name in [(bh, "$B_h$"), (bv, "$B_v$"), (bl, "$B_l$")]:
            angles = numpy.degrees(self.get_angles())
            axes.plot(angles, bfield, label=name)
        ylim = axes.get_ylim()
        for angle in [0.0, 90.0, 180, 270.0, 360.0]:
            axes.plot([angle, angle], ylim, linestyle=":")
        axes.set_ylim(ylim)
        axes.set_xlabel("angle [degree]")
        axes.set_ylabel("B [T]")
        axes.legend()
        return figure

    def plot_fft(self, a_fft):
        """
        Plot the fourier transform
        - a_fft: fourier transform output
        """
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot()
        x_axis = [i for i, z in enumerate(a_fft)]
        axes.semilogy()
        axes.scatter(x_axis, numpy.real(a_fft), label="real", s=2)
        axes.scatter(x_axis, numpy.imag(a_fft), label="imaginary", s=2)
        axes.legend()
        return figure

def default_config():
    config = {
        "working_directory":"lattice/",
        "lattice_filename":"benchmark_lattice-20260216_placement-testing",
        "trajectory_filename":"benchmark_lattice-20260216_placement-testing-trackOrbit.dat",
        "harmonic_analysis_track_id":"ID1", # ID of trajectory
        "harmonic":128, # number of terms in FFT
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

