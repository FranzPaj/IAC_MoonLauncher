'''
Code for producing relevant parameters concerning Mars transfer trajectories (Porkchop plots etc).
'''

# Generic imports
import numpy as np
import os

# Tudat imports
from tudatpy import constants

# Custom imports
from pycode.HelperFunctions import DirectPlanetTransfer


# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()


if __name__ == '__main__':

    ###########################################################################
    # GET PORKCHOP DATA #######################################################
    ###########################################################################

    # Get porkchop slices for Mars transfer

    # Initialise the Direct Transfer class

    sma_arr = (300 + 3390) * 10**3  # Test orbit around Mars with a 300 km altitude
    ecc_arr = 0

    direct_transfer = DirectPlanetTransfer('Earth', 'Mars', sma_arr, ecc_arr)

    # Departure time
    departure_time_density = 1 * constants.JULIAN_DAY  # Map every day
    departure_time_start = 25 * constants.JULIAN_YEAR  # Map from 01/01/2000 onwards
    departure_time_range = 2 * 779.94 * constants.JULIAN_DAY  # 2 Martian synodic periods
    # ToF
    tof_density = 1 * constants.JULIAN_DAY
    tof_minimum = 100 * constants.JULIAN_DAY
    tof_maximum = 700 * constants.JULIAN_DAY
    # Get lists
    departure_time_ls = np.arange(departure_time_start, departure_time_start + departure_time_range, departure_time_density)
    tof_ls = np.arange(tof_minimum, tof_maximum, tof_density)

    # Cover the desired cases
    options = ['earth_launch', 'moon_launch']

    for option in options:

        # Get the option-specific DeltaV
        match option:
            case 'earth_launch':
                earth_sma_dep = (6371 + 300) * 10 ** 3 
                earth_ecc_dep = 0
                # Get the DeltaV
                deltav_matrix = np.zeros((len(tof_ls), len(departure_time_ls)))
                # Evaluate DeltaV
                for i, departure_time in enumerate(departure_time_ls):
                    for j, tof in enumerate(tof_ls):
                        deltav = direct_transfer.get_deltav_earth_launch(departure_time, tof, earth_sma_dep, earth_ecc_dep)
                        deltav_matrix[j, i] = deltav
            case 'moon_launch':
                moon_sma_dep = (3390 + 100) * 10 ** 3
                moon_ecc_dep = 0
                # Get the DeltaV
                deltav_matrix = np.zeros((len(tof_ls), len(departure_time_ls)))
                # Evaluate DeltaV
                for i, departure_time in enumerate(departure_time_ls):
                    for j, tof in enumerate(tof_ls):
                        deltav = direct_transfer.get_deltav_moon_launch(departure_time, tof, moon_sma_dep, moon_ecc_dep)
                        deltav_matrix[j, i] = deltav

        # Store the data
        output_path = os.path.join(current_dir, 'output', 'mars_transfer', option + '.txt')
        np.savetxt(output_path, deltav_matrix)

    # Store search space
    # Departure time
    output_path = os.path.join(current_dir, 'output', 'mars_transfer', 'departure_time_range.txt')
    np.savetxt(output_path, departure_time_ls)
    # ToF
    output_path = os.path.join(current_dir, 'output', 'mars_transfer', 'tof_range.txt')
    np.savetxt(output_path, tof_ls)



