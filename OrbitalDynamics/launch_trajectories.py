from pycode.HelperFunctions import Orbit, average_radius_dict, gravitational_param_dict, LaunchTrajectory, initial_velocity, initial_launch_angle
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    # Define gloabal and design parameters
    R_moon = average_radius_dict['Moon']
    mu_moon = gravitational_param_dict['Moon']
    h = 100e3

    initial_v = np.sqrt(2 * (mu_moon / R_moon - mu_moon / (2 * R_moon + h)))
    trajectory = LaunchTrajectory('Moon', initial_v, 0)

    V0 = [initial_v]
    deltaV = [trajectory.get_deltav_for_circularization()]
    for angle in range(1, 90):
        initial_v = fsolve(initial_velocity, 1e3, args=(np.radians(angle), h))[0]
        trajectory = LaunchTrajectory('Moon', initial_v, np.radians(angle))

        V0.append(initial_v)
        deltaV.append(trajectory.get_deltav_for_circularization())


    plt.plot(range(90), deltaV)
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Delta-V [m/s]')
    plt.savefig(os.path.join('plots', 'delta_v_vs_gamma.png'))
    plt.show()
