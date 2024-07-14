"""
Helper functions and classes that may be useful in other programs
"""

# General imports
import numpy as np
from scipy.optimize import fsolve
from functools import partial

# Tudat imports
from tudatpy.interface import spice
from tudatpy.astro.element_conversion import cartesian_to_keplerian

###################################################
####### LOAD USEFUL CELESTIAL PARAMETERS ###########
###################################################

# Load spice kernels
spice.load_standard_kernels()

# Gravitational parameters
mu_sun = spice.get_body_gravitational_parameter('Sun')
mu_moon = spice.get_body_gravitational_parameter('Moon')
mu_earth = spice.get_body_gravitational_parameter('Earth')
gravitational_param_dict = {
    'Sun': mu_sun,
    'Earth': mu_earth,
    'Moon': mu_moon,
}
# Average radius
R_sun = spice.get_average_radius('Sun')
R_moon = spice.get_average_radius('Moon')
R_earth = spice.get_average_radius('Earth')
average_radius_dict = {
    'Sun': R_sun,
    'Earth': R_earth,
    'Moon': R_moon,
}

sma_dict = {
    'Mercury': 57.9e9,
    'Venus': 108.2e9,
    'Earth': 149.6e9,
    'Mars': 227.9e9,
    'Jupiter': 778.6e9,
    'Saturn': 1433.5e9,
    'Uranus': 2872.5e9,
    'Neptune': 4495.1e9,
}

# Approximate orbit semi-major axes at t = 0 (J2000)
# Moon around Earth
cartesian_state_moon = spice.get_body_cartesian_state_at_epoch('Moon', 'Earth', 'ECLIPJ2000', 'NONE', 0)
keplerian_state_moon = cartesian_to_keplerian(cartesian_state_moon, mu_earth)
sma_moon = keplerian_state_moon[0]
# Earth around Sun
cartesian_state_earth = spice.get_body_cartesian_state_at_epoch('Earth', 'Sun', 'ECLIPJ2000', 'NONE', 0)
keplerian_state_earth = cartesian_to_keplerian(cartesian_state_earth, mu_sun)
sma_earth = keplerian_state_earth[0]


##################################################
############## CUSTOM CLASSES ####################
##################################################
# TODO: Document classes and methods
class Orbit:
    '''
    DESCRIPTIONM TBD
    '''

    def __init__(self,
                 orbited_body: str,
                 semi_major_axis: float,
                 eccentricity: float = 0,
                 inclination: float = np.pi / 2,
                 omega: float = 0,  # Longitude of the ascenscending node
                 raan: float = 0,  # Right ascension of the ascending node
                 theta: float = 0):
        # Attribute Keplerian orbital parameters
        self.orbited_body = orbited_body
        self.sma = semi_major_axis
        self.ecc = eccentricity
        self.incl = inclination
        self.omega = omega
        self.raan = raan
        self.theta = theta

        # Attribute relevant physical environment parameters
        self.mu = gravitational_param_dict[self.orbited_body]
        self.R = average_radius_dict[self.orbited_body]

        # Useful orbit variables
        self.rp = self.sma * (1 - self.ecc)  # Periapsis radius
        self.ra = self.sma * (1 + self.ecc)  # Apoapsis radius
        self.vp = np.sqrt(self.mu * (2 / self.rp - 1 / self.sma))  # Velocity at periapsis
        self.va = np.sqrt(self.mu * (2 / self.ra - 1 / self.sma))  # Velocity at apoapsis

        self.p = self.sma * (1 - self.ecc ** 2)  # Semi-latus rectum
        self.h = self.va * self.ra  # Specific angular momentum

    def get_velocity_for_escape(self, escape_velocity: float = 0):
        '''
        Description
        '''

        rp = self.sma * (1 - self.ecc)  # Get radius of periapsis for ideal impulsive thrust
        velocity_at_periapsis = np.sqrt(escape_velocity ** 2 + 2 * self.mu / self.rp)

        return velocity_at_periapsis

    def get_deltav_for_escape(self, escape_velocity: float = 0):
        # Define nominal velocity at periapsis
        velocity_at_periapsis = np.sqrt(self.mu * (2 / self.rp - 1 / self.sma))
        # Required velocity at periapsis for escape
        required_velocity_at_periapsis = self.get_velocity_for_escape(escape_velocity)
        # Needed DeltaV
        deltav = required_velocity_at_periapsis - velocity_at_periapsis

        return deltav

    def get_deltav_for_circularization(self):
        """
        Function for calculating the delta-v required to circularize at apogee
        """
        vc = np.sqrt(self.mu / self.ra)
        return vc - self.va


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
        self.ecc = np.sqrt(1 - self.R * self.V0 ** 2 / self.mu * (2 - self.R * self.V0 ** 2 / self.mu) * np.cos(
            self.launch_angle) ** 2)

        self.sma = self.p / (1 - self.ecc ** 2)
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
        self.ecc = np.sqrt(1 - self.R * self.V0 ** 2 / self.mu * (2 - self.R * self.V0 ** 2 / self.mu) * np.cos(
            self.launch_angle) ** 2)

        self.sma = self.p / (1 - self.ecc ** 2)
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

    def get_initial_launch_angle_for_altitude(self, h: float = 100e3):
        """
        Calculate initial launch angle to reach an altitude h with a particular initial velocity at the Moon
        :param h: float --> altitude of the target orbit.
        :return: launch angle - float.
        """
        initial_gamma = partial(initial_launch_angle, V0=self.V0, h0=h)
        angle, _, flag, _ = fsolve(initial_gamma, np.radians(45), full_output=True)
        if flag == 1:
            return angle[0]
        else:
            raise ValueError('No solution found for the initial launch angle, try a different initial speed')

    # TODO: Implement interation and optimization for energy and delta-v calculations
    # TODO: Design and implement gravity turn


class Transfer:
    def __init__(self, parking_orbit: Orbit, target_orbit: Orbit, orbited_body_during_transfer: str = 'Sun',
                 r1: float = None, r2: float = None):
        self.parking_orbit = parking_orbit
        self.target_orbit = target_orbit
        self.transfer_body = orbited_body_during_transfer

        # TODO: Implement 2 body transfer
        if orbited_body_during_transfer != 'Sun':
            if r1 is None or r2 is None:
                raise ValueError('Please provide the initial (r1) and final (r2) distances for a 2-body transfer')
            if (orbited_body_during_transfer != parking_orbit.orbited_body or
                    orbited_body_during_transfer != target_orbit.orbited_body):
                raise ValueError(
                    'The transfer body should be either the departure or target body for a transfer not around the Sun')

        if r1 is not None:
            self.r1 = sma_dict[parking_orbit.orbited_body]
        else:
            self.r1 = r1

        if r2 is not None:
            self.r2 = sma_dict[target_orbit.orbited_body]
        else:
            self.r2 = r2

        # Calculate transfer orbit between bodies
        sma = np.linalg.norm(self.r1 - self.r2) / 2  # Ignore altitudes for sma calculation
        ecc = np.linalg.norm(self.r1) + parking_orbit.ra / sma - 1  # ra = a(1 + e)
        self.transfer_orbit = Orbit(orbited_body_during_transfer, sma, ecc)

    def get_transfer_delta_V(self):
        if self.transfer_body == 'Sun':
            return self._three_body_delta_V()
        else:
            return self._two_body_delta_V()

    def _two_body_delta_V(self):
        """
            A first-order approximation of the delta-V for the transfer between 2 bodies
            Assumptions:
                - The orbit is coplanar
                - Orbits are Keplerian within the sphere of influence of a body
        """
        # TODO: Finish implementing this function
        mu0 = gravitational_param_dict[self.transfer_body]
        v_moon = np.sqrt(gravitational_param_dict[self.transfer_body] / self.transfer_orbit.ra)
        if self.target_orbit.orbited_body == self.transfer_body:
            # Moon-Earth transfer
            SOI_moon = self.transfer_orbit.ra * (gravitational_param_dict[self.parking_orbit.orbited_body] / mu0)**(2/5)

            theta_SOI = np.arccos((self.transfer_orbit.p / self.transfer_orbit.sma - 1) / self.transfer_orbit.ecc)  # Where the fuck did this ccome from??
            r_SOI = np.sqrt(self.transfer_orbit.ra ** 2 + SOI_moon ** 2 - 2 * self.transfer_orbit.ra * SOI_moon * np.cos(np.pi - theta_SOI))
            v_SOI = np.sqrt(2 * (mu0 / r_SOI - mu0 / (2 * self.transfer_orbit.sma)))
            gamma_SOI = np.arccos(self.transfer_orbit.h / (r_SOI * v_SOI))

            v_inf = v_SOI * np.array([[np.cos(gamma_SOI)], [np.sin(gamma_SOI)]]) - np.array([[0], [v_moon]])
            v_inf = np.linalg.norm(v_inf)

            deltaV_departure = np.sqrt(2 * (v_inf ** 2 / 2 + mu0 / self.linalg.norm(self.r1))) - self.parking_orbit.vp
            deltaV_escape = np.abs(self.transfer_orbit.vp - np.sqrt(mu0 / np.linalg.norm(self.r1)))
        else:
            # Earth-Moon transfer
            SOI_moon = self.transfer_orbit.ra * (gravitational_param_dict[self.target_orbit.orbited_body] / mu0)**(2/5)



        return deltaV_escape + deltaV_departure

    def _three_body_delta_V(self):
        """
        A first-order approximation of the delta-V for the transfer between 2 bodies round the Sun
        Assumptions:
            - The orbit is coplanar
            - Orbits are Keplerian within the sphere of influence of a body
            - Gamma is almost 0 at the border of the SOI of the planets
        """

        # TODO: verify and validate function
        # Calculate the velocities of the planets
        v_planet_arrival = np.sqrt(gravitational_param_dict[self.transfer_body] / self.target_orbit.ra)
        v_planet_departure = np.sqrt(gravitational_param_dict[self.transfer_body] / self.parking_orbit.ra)

        # Arrival parameters
        v2 = self.transfer_orbit.va - v_planet_arrival
        v_inf_2 = np.sqrt(v2 ** 2 - 2 * gravitational_param_dict[self.target_orbit.orbited_body] / self.target_orbit.rp)

        delta_V = v2 - self.target_orbit.vp  # Impulse to circularize at arrival

        # Transfer parameters
        v_SOI_2 = v_inf_2 + v_planet_arrival
        v_SOI_1 = np.sqrt(2 * (v_SOI_2 ** 2 / 2 - gravitational_param_dict[self.transfer_body] * (
                1 / self.transfer_orbit.rp - 1 / self.transfer_orbit.ra)))

        # Arrival parameters
        v_inf_1 = v_SOI_1 - v_planet_departure
        v1 = np.sqrt(v_inf_1 + 2 * gravitational_param_dict[self.parking_orbit.orbited_body] / self.parking_orbit.rp)

        delta_V += v1 - self.parking_orbit.vp  # Impulse to depart
        return delta_V


########################################
######### CUSTOM FUNCTIONS #############
########################################

# TODO: Do we need this function?
def get_sma_from_altitude(orbited_body: str,
                          altitude: float):
    '''
    Description TBD
    '''

    average_radius = average_radius_dict[orbited_body]  # Get average radius of orbited body
    sma = average_radius + altitude  # Get semi-major axis of equivalent circular orbit

    return sma


def initial_velocity(V0: float, gamma: float = 0, h: float = 100e3):
    """
    Calculate initial velocity to reach an altitude h with a particular launch angle at the Moon
    :param V0: float --> initial velocity.
    :param gamma: float --> launch angle in radians.
    :param h: float --> altitude of the circular orbit.
    :return: float --> initial velocity.
    """

    p = (R_moon * V0 * np.cos(gamma)) ** 2 / mu_moon
    e = np.sqrt(1 - R_moon * V0 ** 2 / mu_moon * (2 - R_moon * V0 ** 2 / mu_moon) * np.cos(gamma) ** 2)

    ra = p / (1 - e)
    ra_target = R_moon + h
    return ra - ra_target


def initial_launch_angle(gamma: float, V0: float = 1754, h0: float = 100e3):
    return initial_velocity(V0, gamma, h0)
