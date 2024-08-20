'''
Code for analysing the capabilities of different launchers and scenarios. It covers the differences in P/L that can be carries to a given
destination, as well as the Launch window tradeoffs that correspond to different launch methods (Earth launch, Earth refuelling, 
Moon launch, Moon refuelling).
'''

# Generic imports
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

# Tudat imports
from tudatpy import constants

# Custom imports
from OrbitalDynamics.pycode.HelperFunctions import DirectPlanetTransfer, Launcher
from OrbitalDynamics.pycode import PlotGenerator as PlotGen

# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()


if __name__ == '__main__':

    ###########################################################################
    # READ USEFUL DATA ########################################################
    ###########################################################################

    # Get the directory for the stored yaml data
    yaml_dir = os.path.join(current_dir, 'OrbitalDynamics', 'input', 'launcher_data')
    # Get the paths for the specific launchers
    starship_path = os.path.join(yaml_dir, 'starship.yaml')

    starship = Launcher(starship_path)

    starship_refuelled_deltav_limit = starship.get_available_refuelled_deltav(4000)

    print(starship_refuelled_deltav_limit)


    # -----------------------------------------------------------------------------------------------

    ###########################################################################
    # MARS CASE ###############################################################
    ###########################################################################

    print('-------------------------------------')
    print('Mars case study')

    ###########################################################################
    # PLOT FIGURES FOR MARS TRANSFER ##########################################
    ###########################################################################

    data_dir = os.path.join(current_dir, 'OrbitalDynamics', 'output', 'mars_transfer')

    # Load the relevant data
    departure_time_ls = np.loadtxt(os.path.join(data_dir, 'departure_time_range.txt'))
    tof_ls = np.loadtxt(os.path.join(data_dir, 'tof_range.txt'))
    deltav_matrix_earth = np.loadtxt(os.path.join(data_dir, 'earth_launch.txt'))
    deltav_matrix_moon = np.loadtxt(os.path.join(data_dir, 'moon_launch.txt'))

    min_deltav_earth = np.min(deltav_matrix_earth)
    min_deltav_moon = np.min(deltav_matrix_moon)
    print('Minimum DeltaV needed for Mars transfer')
    print('Earth:', min_deltav_earth/1000, 'km/s', '| Moon:', min_deltav_moon/1000, 'km/s')

    starship_possible_refuelled_earth_pl = starship.get_possible_refuelled_pl(min_deltav_earth)
    starship_possible_refuelled_moon_pl = starship.get_possible_refuelled_pl(min_deltav_moon)
    print('Possible P/L transportable via starship')
    print('Earth refuel:', starship_possible_refuelled_earth_pl / 1000, 't', '| Moon refuel:', starship_possible_refuelled_moon_pl / 1000, 't')

    fig1 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_earth)

    fig2 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_moon)

    res_dict = {
        'earth_refuel':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_earth,
            'deltav_cutoff': starship_refuelled_deltav_limit,
            'color': 'green',
            'linestyle': '-',
            'label': 'Earth refuelling',
        },
        'moon_refuel':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_moon,
            'deltav_cutoff': starship_refuelled_deltav_limit,
            'color': 'grey',
            'linestyle': '--',
            'label': 'Moon refuelling',
        },
    }

    fig3 = PlotGen.feasibility_comparison(res_dict)

    plt.show()