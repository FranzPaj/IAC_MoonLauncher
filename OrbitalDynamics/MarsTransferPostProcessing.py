'''
Code for post-processing relevant parameters concerning Mars transfer trajectories (Porkchop plots etc).
'''

# Generic imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom imports
from pycode import PlotGenerator as PlotGen


# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()


if __name__ == '__main__':

    data_dir = os.path.join(current_dir, 'output', 'mars_transfer')

    # Load the relevant data
    departure_time_ls = np.loadtxt(os.path.join(data_dir, 'departure_time_range.txt'))
    tof_ls = np.loadtxt(os.path.join(data_dir, 'tof_range.txt'))
    deltav_matrix_earth = np.loadtxt(os.path.join(data_dir, 'earth_launch.txt'))
    deltav_matrix_moon = np.loadtxt(os.path.join(data_dir, 'moon_launch.txt'))

    fig1 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_earth, deltav_cutoff=12)

    fig2 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_moon, deltav_cutoff=14)

    plt.show()


