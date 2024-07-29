from pycode.HelperFunctions import Orbit, Transfer
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    # Test for direct Hohmann transfer

    parking_orbit = Orbit('Earth', (300 + 6371) * 10**3, 0)
    target_orbit = Orbit('Mars', (300 + 3390) * 10**3, 0)

    direct_transfer = Transfer(parking_orbit, target_orbit, 'Sun')

    deltav = direct_transfer.get_transfer_delta_V()

    print(deltav)

    

