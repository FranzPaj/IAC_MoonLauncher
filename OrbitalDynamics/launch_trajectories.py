from pycode.HelperFunctions import Orbit, average_radius_dict, gravitational_param_dict
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np


class LaunchTrajectory(Orbit):
    def __init__(self, orbited_body: str, initial_velocity: float = 1703.54, launch_angle: float = 0):

        # Initial conditions
        self.launch_angle = launch_angle
        self.V0 = initial_velocity

        # Attribute relevant physical environment parameters
        self.mu = gravitational_param_dict[orbited_body]
        self.R = average_radius_dict[orbited_body]

        # Calculate resulting orbit
        self.p = (self.R * self.V0 * np.cos(self.launch_angle)) ** 2 / self.mu
        self.ecc = np.sqrt(1 - self.R * self.V0**2 / self.mu * (2 - self.R * self.V0**2 / self.mu) * np.cos(self.launch_angle)**2)

        self.sma = self.p / (1 - self.ecc**2)
        self.theta = np.arccos((self.p / self.R - 1) / self.ecc)

        super().__init__(orbited_body=orbited_body, semi_major_axis=self.sma, eccentricity=self.ecc, theta=self.theta)

    def reinitialize_paramters(self, new_launch_angle: float, new_initial_velocity: float):
        """
        Function for resetting the parameters of the orbit
        """
        self.launch_angle = new_launch_angle
        self.V0 = new_initial_velocity

        # Calculate resulting orbit
        self.p = (self.R * self.V0 * np.cos(self.launch_angle)) ** 2 / self.mu
        self.ecc = np.sqrt(1 - self.R * self.V0**2 / self.mu * (2 - self.R * self.V0**2 / self.mu) * np.cos(self.launch_angle)**2)

        self.sma = self.p / (1 - self.ecc**2)
        self.theta = np.arccos((self.p / self.R - 1) / self.ecc)

        super().__init__(self.orbited_body, self.sma, self.ecc, theta=self.theta)

    def get_max_altitude(self):
        return self.ra - self.R

    def get_initial_velocity_for_altitude(self, h: float = 100e3):
        """
        Calculate initial velocity to reach an altitude h with a particular launch angle at the Moon
        :param h: float --> altitude of the target orbit.
        :return: float --> initial velocity.
        """
        V0 = fsolve(initial_velocity, self.V0, args=(self.launch_angle, h))[0]
        return V0

    # TODO: Implement interation and optimization for energy and delta-v calculations
    # TODO: Design and implement gravity turn


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
    h = 100e3

    V0 = [np.sqrt(2 * (mu_moon / R_moon - mu_moon / (2 * R_moon + h)))]
    for angle in range(1, 90):
        V0.append(fsolve(initial_velocity, 1e3, args=(np.radians(angle), h))[0])

    # plt.plot(np.arange(0, 90, 1), V0)
    # plt.xlabel('Launch angle [deg]')
    # plt.ylabel('Initial velocity [m/s]')
    # plt.show()

    # Test the LaunchTrajectory class
    initial_V = V0[0]
    launch_angle = np.radians(0)
    trajectory = LaunchTrajectory('Moon', initial_V, launch_angle)

    print('Maximum altitude = ', trajectory.get_max_altitude() / 1e3, ' km')
    print('Delta_V for circularization = ', trajectory.get_deltav_for_circularization() / 1e3, ' km/s')
    print('Necessary velocity to reach 150km = ', trajectory.get_initial_velocity_for_altitude(150e3) / 1e3, ' km')

