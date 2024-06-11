# from pycode.HelperFunctions import Orbit, R_moon, mu_moon
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np


def initial_velocity(V0: float, gamma: float = 0, h: float = 100e3):
    """
    Calculate initial velocity to reach an altitude h with a particular launch angle at the Moon
    :param V0: float --> initial velocity.
    :param gamma: float --> launch angle in radians.
    :param h: float --> altitude of the circular orbit.
    :return: float --> initial velocity.
    """

    p = (R_moon * V0 * np.cos(gamma)) ** 2 / mu_moon
    e = np.sqrt(1 - R_moon * V0**2 / mu_moon * (2 - R_moon * V0**2 / mu_moon) * np.cos(gamma)**2)

    ra = p / (1 - e)
    ra_target = R_moon + h
    return ra - ra_target



if __name__ == '__main__':

    R_moon = 1737.4e3
    mu_moon = 4.9048695e12

    V0 = []
    gamma = np.radians(np.linspace(1, 89, 88))
    for angle in gamma:
        V0.append(fsolve(initial_velocity, 1e3, args=(angle, 100e3)))

    plt.plot(np.degrees(gamma), V0)
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Initial velocity [m/s]')
    plt.savefig('Plots\\Initial_velocity_vs_gamma.pdf')
    plt.show()

