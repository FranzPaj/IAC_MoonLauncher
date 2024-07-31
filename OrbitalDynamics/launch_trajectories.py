from pycode.HelperFunctions import Orbit, Transfer, DirectPlanetTransfer
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tudatpy import constants

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

if __name__ == '__main__':

    #### Test for direct Hohmann transfer

    hohmann_test()

    #### Test for Lambert Targeter

    lambert_test()




