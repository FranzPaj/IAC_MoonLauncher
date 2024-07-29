from pycode.HelperFunctions import Orbit, Transfer, PlanetDirectTransfer
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os
from tudatpy import constants



if __name__ == '__main__':

    # Test for direct Hohmann transfer

    sma_arr = (300 + 3390) * 10**3  # Test orbit around Mars with a 300 km altitude
    ecc_arr = 0

    parking_orbit = Orbit('Earth', (300 + 6371) * 10**3, 0)
    target_orbit = Orbit('Mars', sma_arr, ecc_arr)

    direct_transfer = Transfer(parking_orbit, target_orbit, 'Sun')

    deltav = direct_transfer.get_transfer_delta_V()

    # Test for Lambert Targeter

    direct_transfer = PlanetDirectTransfer('Earth', 'Mars', sma_arr, ecc_arr,)

    tup = direct_transfer.get_transfer_parameters(22 * constants.JULIAN_YEAR, 250 * constants.JULIAN_DAY)

    print(tup)


