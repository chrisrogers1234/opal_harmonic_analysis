import os

import numpy
import pandas
import matplotlib
import matplotlib.pyplot


class PlotDumpEMFields:
    def __init__(self, config):
        self.emfield_columns = ["x", "y", "z", "t", "bx", "by", "bz", "ex", "ey", "ez"]
        self.config = config

    def do_plots(self, trajectory=None):
        self.load_em_fields()
        self.plot_em_fields(trajectory)

    def load_em_fields(self):
        emfield_filename = self.config['emfield_filename']
        if emfield_filename == None:
            return
        self.emfield = pandas.read_csv(emfield_filename, sep=" ",
                      header=0, names=self.emfield_columns, skiprows=12)
        if self.config["verbose"] > 2:
            print(f"Loaded {self.emfield.shape[0]} rows from emfield map")

    def bins(self, values):
        values = sorted(list(set(values)))
        a_min = values[0]+(values[1]-values[0])/2
        a_max = values[-1]+(values[-1]-values[-2])/2
        n_bins = len(values)+1
        return numpy.linspace(a_min, a_max, n_bins)

    def plot_em_fields(self, trajectory=None):
        x_bins = self.bins(self.emfield["x"])
        y_bins = self.bins(self.emfield["y"])
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.hist2d(self.emfield["x"], self.emfield["y"], weights=self.emfield["bz"], bins=[x_bins, y_bins])
        if trajectory is not None:
            axes.plot(trajectory["x"], trajectory["y"])

        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        figure.savefig("cartesian_bz.png")


def default_config():
    config = {
        "working_directory":"lattice/isis1",
        "emfield_filename":"data/CoarseField_0.dat",
        "verbose":100,
    }
    return config

def main():
    config = default_config()
    os.chdir(config["working_directory"])
    plotter = PlotDumpEMFields(config)
    plotter.do_plots()


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
