from OrbitalDynamics.pycode.HelperFunctions import Orbit, Transfer, DirectPlanetTransfer, LaunchTrajectory
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tudatpy import constants
from tqdm import tqdm

###########################################################################
# TEST FUNCTIONS ##########################################################
###########################################################################

def lambert_test():

    sma_arr = (300 + 3390) * 10**3  # Test orbit around Mars with a 300 km altitude
    ecc_arr = 0

    direct_transfer = DirectPlanetTransfer('Earth', 'Mars', sma_arr, ecc_arr,)

    # circa a Hohmann transfer
    departure_epoch = 22 * constants.JULIAN_YEAR + (9*30 + 2 + 4) * constants.JULIAN_DAY
    tof = 259 * constants.JULIAN_DAY
    earth_sma_dep = (6371 + 300) * 10 ** 3 
    earth_ecc_dep = 0

    moon_sma_dep = (3390 + 100) * 10 ** 3
    moon_ecc_dep = 0

    earth_deltav = direct_transfer.get_deltav_earth_launch(departure_epoch, tof, earth_sma_dep, earth_ecc_dep)
    moon_deltav = direct_transfer.get_deltav_moon_launch(departure_epoch, tof, moon_sma_dep, moon_ecc_dep)

    print('DeltaV needed if launching from the Earth:', earth_deltav / 10**3, 'km/s')
    print('DeltaV needed if launching from the Moon:', moon_deltav / 10**3, 'km/s')

    return

def hohmann_test():

    sma_arr = (300 + 3390) * 10**3  # Test orbit around Mars with a 300 km altitude
    ecc_arr = 0

    parking_orbit = Orbit('Earth', (300 + 6371) * 10**3, 0)
    target_orbit = Orbit('Mars', sma_arr, ecc_arr)

    direct_transfer = Transfer(parking_orbit, target_orbit, 'Sun')

    deltav = direct_transfer.get_transfer_delta_V()

    return


def moon_launch_optim(plot: bool = False, return_params: bool = False):
    launch_angle = np.radians(np.arange(1, 90, .5))
    initial_v = 1.5e3
    launch_params = []
    delta_v = []
    for angle in tqdm(launch_angle):
        # Define inital orbit
        launch_trajectory = LaunchTrajectory('Moon', initial_velocity=initial_v, launch_angle=angle)

        # Calculate minimum velocity and update
        vel = launch_trajectory.get_initial_velocity_for_altitude(100e3)
        launch_trajectory.reinitialize_paramters(angle, vel)

        # Store results
        delta_v.append(launch_trajectory.get_deltav_for_circularization())
        launch_params.append([angle, vel])

    if plot:
        plt.plot(np.degrees(launch_angle), delta_v)
        plt.xlabel('Launch angle [deg]')
        plt.ylabel('DeltaV [m/s]')
        plt.show()

    return np.array(delta_v) if not return_params else np.array(delta_v), np.array(launch_params)


if __name__ == '__main__':
    #### Test for direct Hohmann transfer
    # hohmann_test()

    #### Test for Lambert Targeter
    # lambert_test()

    #### Moon launch optimization
    deltaV, launch_params = moon_launch_optim(return_params=True)

    # Launch parameter results
    angle, velocity = np.array(launch_params).T
    plt.plot(np.degrees(angle), velocity)
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Initial velocity [m/s]')
    plt.title('Initial velocity vs Launch angle for Moon launch')
    plt.show()

    # Mass ratio results
    deltaV = np.array(deltaV)
    baseline_ratio = np.exp(1.8e3 / (300 * 9.81)) - 1
    alternative_ratio = np.exp(deltaV / (300 * 9.81)) - 1
    print('Baseline ratio:', baseline_ratio)

    plt.plot(np.arange(1, 90, .5), alternative_ratio, label='alternative')
    plt.axhline(y=baseline_ratio, color='r', linestyle='--', label='baseline')
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Mass ratio')
    plt.title('Mp / (Mu + Mc) vs Launch angle for Isp = 300s')
    plt.legend()
    plt.show()




