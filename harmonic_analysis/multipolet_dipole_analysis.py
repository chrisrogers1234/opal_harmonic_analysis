import numpy
import matplotlib
import matplotlib.pyplot
import math
import pyopal.elements.multipolet

import tracking_analysis
import harmonic_analysis

class DipoleAnalysis(pyopal.objects.minimal_runner.MinimalRunner):
    def __init__(self):
        super().__init__()
        self.tmp_dir = "lattice/isis1_dipole"
        self.harmonics = harmonic_analysis.HarmonicAnalysis(harmonic_analysis.isis1_config())
        self.simulation = None
        self.ha_step = None
        self.r0 = 100
        self.length = 4.4
        self.angle_deg = 36
        self.number_of_body_points = 400
        self.number_of_end_points = 40
        self.multipole_order = 1 # 1 or 2
        self.x_lim = [4.0, 5.0]
        self.h_lim = [-0.2, 0.2]
        self.plot_suffix = ""


    def setup(self):
        self.simulation = tracking_analysis.Simulation()

    def make_element_iterable(self):
        """
        Add in a ISIS dipole a la Jon's lattice
        """
        multipole = pyopal.elements.multipolet.MultipoleT()
        #sp8_dip1 : MULTIPOLET, L=4.4, HAPERT=2, VAPERT=2, MAXFORDER=4,
        #           TP={-0.175827926,0.08351019049,0,0,0}, LFRINGE=0.09, RFRINGE=0.09,
        #           ROTATION=0, ANGLE=0.628319, EANGLE=0, MAXXORDER=20, VARRADIUS=FALSE,
        #           BBLENGTH=6, SCALING_MODEL="MagnetAmplitude";
        multipole.set_attributes(
            horizontal_aperture=2,
            vertical_aperture=2,
            left_fringe=0.09,
            right_fringe=0.09,
            length=self.length,
            angle=math.radians(self.angle_deg),
            rotation=0.0,
            maximum_f_order=4,
            maximum_x_order=20,
            variable_radius=False,
            bounding_box_length=6.0,
            t_p=[-0.175827926,0.08351019049][:self.multipole_order]
        )
        return [multipole]

    def postprocess(self):
        self.plot_field()
        self.plot_harmonics()

    def get_trajectory(self, r_offset, yaw_offset):
        """
        Get phase space vectors in the coordinate system of the magnet.

        - number_of_body_points: number of points between 0 and self.length,
        evenly spread.
        - number_of_end_points: number of points before and after the body;
        using spacing set by the body points.
        - r_offset: radius of the points, relative to the dipole centre, will be
        r_curv+r_offset where r_curv is the radius of curvature of the magnet
        centre.
        - yaw_offset: momentum coordinates will point along the magnet centre with
        an additional rotation in horizontal plane by yaw_offset. A 90 degree
        yaw_offset points towards the centre of the bend radius; a -90 degree
        yaw_offset points towards the outside of the dipole directly away from
        the centre.

        In total there will be
            number_of_body_points + 2 * number_of_end_points+1
        returned.

        Returns list of dictionaries where each dictionary is a phase space
        vector with elements "step", "x", "y", "z", "xp", "yp", "zp", "s"
        """
        psv_list = []
        for i in range(-self.number_of_end_points,
                       self.number_of_body_points+self.number_of_end_points+1):
            s = self.length/self.number_of_body_points*i
            if self.angle_deg == 0:
                psv = {"step":i, "id":1, "x":self.r0, "y":s, "z":0.0, "xp":0, "yp":1, "zp":0.0, "s":s}
                psv_list.append(psv)
                continue
            r_curv = self.length/math.radians(self.angle_deg)+r_offset
            theta = math.radians(self.angle_deg)*i/self.number_of_body_points
            x = self.r0+r_curv*(math.cos(theta)-1)
            y = r_curv*math.sin(theta)
            xp = -math.sin(theta+yaw_offset)
            yp = math.cos(theta+yaw_offset)
            psv = {"step":i, "id":1, "x":x, "y":y, "z":0.0, "xp":xp, "yp":yp, "zp":0.0, "s":s}
            psv_list.append(psv)
        return psv_list


    def plot_harmonics(self):
        fft_r_list, fft_l_list, fft_phi_list, s_list = [], [], [], []
        psv_list = self.get_trajectory(0.0, 0.0)
        for psv in psv_list:
            fft_r, fft_phi, fft_l = self.harmonics.analyse_one_step(psv, False)
            fft_r_list.append(fft_r)
            fft_phi_list.append(fft_phi)
            fft_l_list.append(fft_l)
            s_list.append(psv["s"])
            print(f"{psv["s"]:8.4g} {psv["x"]:8.4g} {psv["y"]:8.4g}")
        fft_r_list = numpy.array(fft_r_list).transpose()
        fft_phi_list = numpy.array(fft_phi_list).transpose()
        fft_l_list = numpy.array(fft_l_list).transpose()
        for fft_list, name in [(fft_r_list, "r"), (fft_l_list, "l"), (fft_phi_list, "phi")]:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot()
            axes.plot([], [], label="Real", c="black")
            axes.plot([], [], label="Imag", linestyle="--", c="black")
            for i in range(6):
                scale = 0.1**int(i/2)
                fft = numpy.real(fft_list[i, :])*scale
                label = f"B$_{{{name}}}$ h={i} [{scale:4.3g} T"
                if i > 0:
                    label +=  f"m$^{{-{i}}}$]"
                else:
                    label += "]"
                plot = axes.plot(s_list, fft, label=label)[0]
                fft = numpy.imag(fft_list[i, :])*scale
                axes.plot(s_list, fft, linestyle="--", c=plot._color)

            axes.set_xlim(self.x_lim)
            axes.set_ylim(self.h_lim)
            axes.set_xlabel("s [m]")
            axes.legend()
            figure.savefig(f"isis_dipole_ha_{name}{self.plot_suffix}.png")

    def plot_harmonics_2d(self):
        fft_r_list, fft_l_list, fft_phi_list, s_list = [], [], [], []
        psv_list = self.get_trajectory(0.0, 0.0)
        for psv in psv_list:
            fft_r, fft_phi, fft_l = self.harmonics.analyse_one_step(psv, False)
            fft_r_list.append(fft_r)
            fft_phi_list.append(fft_phi)
            fft_l_list.append(fft_l)
            s_list.append(psv["s"])
            print(f"{psv["s"]:8.4g} {psv["x"]:8.4g} {psv["y"]:8.4g}")
        fft_r_list = numpy.array(fft_r_list).transpose()
        fft_phi_list = numpy.array(fft_phi_list).transpose()
        fft_l_list = numpy.array(fft_l_list).transpose()
        for fft_list, name in [(fft_r_list, "r"), (fft_l_list, "l"), (fft_phi_list, "phi")]:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot()
            axes.plot([], [], label="Real", c="black")
            axes.plot([], [], label="Imag", linestyle="--", c="black")
            for i in range(6):
                scale = 0.1**int(i/2)
                fft = numpy.real(fft_list[i, :])*scale
                label = f"B$_{{{name}}}$ h={i} [{scale:4.3g} T"
                if i > 0:
                    label +=  f"m$^{{-{i}}}$]"
                else:
                    label += "]"
                plot = axes.plot(s_list, fft, label=label)[0]
                fft = numpy.imag(fft_list[i, :])*scale
                axes.plot(s_list, fft, linestyle="--", c=plot._color)

            axes.set_xlim(self.x_lim)
            axes.set_ylim(self.h_lim)
            axes.set_xlabel("s [m]")
            axes.legend()
            figure.savefig(f"isis_dipole_ha_{name}{self.plot_suffix}.png")


    def plot_field(self):
        x_list, y_list, b_list = [], [], []
        r0 = self.r0
        for xi in range(-100, 101):
            for yi in range(-100, 501):
                point = (r0+xi*2e-2, yi*1e-2, 0.0, 0.0)
                x_list.append(point[0])
                y_list.append(point[1])
                field = pyopal.objects.field.get_field_value(*point)
                bfield = (field[1]**2+field[2]**2+field[3]**2)**0.5
                b_list.append(bfield)

        psv_list = self.get_trajectory(0.0, 0.0)
        fig = matplotlib.pyplot.figure()
        axes = fig.add_subplot()
        axes.hist2d(x_list, y_list, weights=b_list, bins=[201, 601])
        axes.plot([psv["x"] for psv in psv_list], [psv["y"] for psv in psv_list])
        fig.axes[0].set_xlabel("x [m]")
        fig.axes[0].set_ylabel("y [m]")
        fig.axes[0].set_title("Total B [T]")
        fig.savefig(f"bfield_isis_dipole{self.plot_suffix}.png")

    def save(self, fft_r, fft_phi, fft_l):
        with open(self.harmonics_file_name, "w") as fout:
            serialisation = {}
            serialisation["fft_r_re"] = numpy.real(fft_r).tolist()
            serialisation["fft_r_im"] = numpy.imag(fft_r).tolist()
            serialisation["fft_phi_re"] = numpy.real(fft_phi).tolist()
            serialisation["fft_phi_im"] = numpy.imag(fft_phi).tolist()
            serialisation["fft_l_re"] = numpy.real(fft_l).tolist()
            serialisation["fft_l_im"] = numpy.imag(fft_l).tolist()
            fout.write(json.dumps(serialisation))

    def load(self):
        with open(self.harmonics_file_name) as fin:
            serialisation = json.loads(fin.read())
            fft_r = numpy.array(serialisation["fft_r_re"])+ \
                    numpy.array(serialisation["fft_r_im"])*1j
            fft_phi = numpy.array(serialisation["fft_phi_re"])+ \
                    numpy.array(serialisation["fft_phi_im"])*1j
            fft_l = numpy.array(serialisation["fft_l_re"])+ \
                    numpy.array(serialisation["fft_l_im"])*1j
            return fft_r, fft_phi, fft_l


def main():
    dipole_analysis = DipoleAnalysis()
    dipole_analysis.execute_fork()

if __name__ == "__main__":
    main()
