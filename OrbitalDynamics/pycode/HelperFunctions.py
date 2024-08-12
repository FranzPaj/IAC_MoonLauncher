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
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import environment
from tudatpy.astro import two_body_dynamics, element_conversion

# Custom imports
from OrbitalDynamics.pycode.CustomConstants import gravitational_param_dict, average_radius_dict, sma_dict

# Activate available debugging prints
debug_flag = False


###########################################################################
# CUSTOM CLASSES ##########################################################
###########################################################################

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
    """
    Subclass meant to be used for the launch trajectory when exiting the kinetic launch system
    """
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
        :return: float --> initial velocity or np.nan if no solution is found.
        """
        try:
            V0 = fsolve(initial_velocity, self.V0, args=(self.launch_angle, h))[0]
        except (RuntimeError, RuntimeWarning):
            print(f'RuntimeWarning for angle = {np.degrees(self.launch_angle)} and velocity = {self.V0}')
            V0 = np.nan
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




class Transfer:
    """
        A class to represent the 2D transfer between two orbits around a celestial body.

        Attributes
        ----------
        parking_orbit : Orbit
            The initial orbit of the spacecraft.
        target_orbit : Orbit
            The target orbit the spacecraft aims to reach.
        transfer_body : str
            The body the spacecraft will orbit during the transfer.

        Methods
        -------
        get_transfer_delta_V(full_output=False):
            Calculates the total delta-V required for the transfer.
        _two_body_delta_V(full_output=False):
            Calculates the delta-V for a transfer between two bodies.
        _three_body_delta_V(full_output=False):
            Calculates the delta-V for a transfer between two bodies around the Sun.
    """

    def __init__(self, parking_orbit: Orbit, target_orbit: Orbit, orbited_body_during_transfer: str = 'Sun',
                 r1: float = None, r2: float = None):
        self.parking_orbit = parking_orbit
        self.target_orbit = target_orbit
        self.transfer_body = orbited_body_during_transfer

        if r1 is None:
            if parking_orbit.orbited_body == orbited_body_during_transfer:
                self.r1 = parking_orbit.rp
            else:
                self.r1 = sma_dict[parking_orbit.orbited_body]
        else:
            self.r1 = r1

        if r2 is None:
            if target_orbit.orbited_body == orbited_body_during_transfer:
                self.r2 = target_orbit.rp
            else:
                self.r2 = sma_dict[target_orbit.orbited_body]
        else:
            self.r2 = r2

        # Calculate transfer orbit between bodies
        sma = (self.r1 + self.r2) / 2  # Ignore altitudes for sma calculation
        ecc = max(self.r1, self.r2) / sma - 1  # ra = a(1 + e)
        self.transfer_orbit = Orbit(orbited_body_during_transfer, sma, ecc)

    def get_transfer_delta_V(self, full_output: bool = False):
        # TODO: V&V this function and delete the subsequent 2
        ############################### Calculate the velocities of the planets #######################################
        if self.parking_orbit.orbited_body != self.transfer_orbit.orbited_body:
            v_planet_departure = (
                np.sqrt(gravitational_param_dict[self.transfer_body] / sma_dict[self.parking_orbit.orbited_body]))
        else:
            v_planet_departure = 0
            out_bound = True

        if self.target_orbit.orbited_body != self.transfer_orbit.orbited_body:
            v_planet_arrival = (
                np.sqrt(gravitational_param_dict[self.transfer_body] / sma_dict[self.target_orbit.orbited_body]))
        else:
            v_planet_arrival = 0
            out_bound = False

        if v_planet_arrival != 0 and v_planet_departure != 0:
            if v_planet_arrival < v_planet_departure:
                # Outwards transfer
                out_bound = True
            else:
                # Inwards transfer
                out_bound = False

        ####################################### Departure impulse ####################################################
        if self.parking_orbit.orbited_body != self.transfer_body:
            # Planet-centred departure velocity (v infinity)
            v_inf_1 = self.transfer_orbit.vp - v_planet_departure if out_bound \
                else self.transfer_orbit.va - v_planet_departure

            # Periapsis velocity of the hyperbolic orbit
            v1 = np.sqrt(v_inf_1 ** 2 + 2 * gravitational_param_dict[self.parking_orbit.orbited_body] / self.parking_orbit.rp)

            # Impulse to enter said hyperbolic orbit
            departure_impulse = np.abs(v1 - self.parking_orbit.vp)
        else:
            departure_impulse = np.abs(self.transfer_orbit.vp - self.parking_orbit.vp) if out_bound else\
                np.abs(self.transfer_orbit.va - self.parking_orbit.vp)

        ####################################### Arrival impulse ####################################################
        if self.target_orbit.orbited_body != self.transfer_body:
            # Planet-centred departure velocity (v infinity)
            v_inf_2 = self.transfer_orbit.vp - v_planet_arrival if not out_bound \
                else self.transfer_orbit.va - v_planet_arrival

            # Periapsis velocity of the hyperbolic orbit
            v2 = np.sqrt(v_inf_2 ** 2 + 2 * gravitational_param_dict[
                self.target_orbit.orbited_body] / self.target_orbit.rp)

            # Impulse to enter said hyperbolic orbit
            arrival_impulse = np.abs(v2 - self.target_orbit.vp)
        else:
            arrival_impulse = np.abs(self.transfer_orbit.vp - self.target_orbit.vp) if not out_bound else \
                np.abs(self.transfer_orbit.va - self.target_orbit.vp)

        ######################################### Calculate delta-V ##################################################
        delta_V = departure_impulse + arrival_impulse

        return delta_V if not full_output else (delta_V, departure_impulse, arrival_impulse)


class DirectPlanetTransfer:

    def __init__(self, departure_body: str, target_body: str, sma_tar: str, ecc_tar: str):

        # Attribute instance properties
        self.departure_body = departure_body
        self.target_body = target_body
        self.sma_tar = sma_tar
        self.ecc_tar = ecc_tar

        # Define new instance properties
        self.r_dep = sma_dict[self.departure_body]
        self.r_tar = sma_dict[self.target_body]

        self.bodies = self.create_simulation_bodies()

    def create_simulation_bodies(self) -> environment.SystemOfBodies:

        """
        Creates the body objects required for the simulation, using the
        environment_setup.create_system_of_bodies for natural bodies,
        and manual definition for vehicles

        Parameters
        ----------
        none

        Return
        ------
        Body objects required for the simulation.

        """

        # Create settings for celestial bodies
        bodies_to_create = ['Sun',self.departure_body, self.target_body]
        global_frame_origin = 'Sun'
        global_frame_orientation = 'ECLIPJ2000'
        body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)

        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        return bodies

    def get_lambert_problem_result(
        self,
        departure_epoch: float,
        tof: float ) -> environment.Ephemeris:

        """"
        This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
        a target body (at arrival epoch), with the states of Earth and the target body defined
        by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
        assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

        Parameters
        ----------
        bodies : Body objects defining the physical simulation environment

        target_body : The name (string) of the body to which the Lambert arc is to be computed

        departure_epoch : Epoch at which the departure from Earth's center of mass is to take place

        arrival_epoch : Epoch at which the arrival at he target body's center of mass is to take place

        Return
        ------
        TBD
        """

        # Gravitational parameter of the Sun
        central_body_gravitational_parameter = self.bodies.get_body("Sun").gravitational_parameter

        global_frame_orientation = 'ECLIPJ2000'
        # Set initial and final positions for Lambert targeter
        initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name=self.departure_body,
            observer_body_name="Sun",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=departure_epoch)

        final_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name=self.target_body,
            observer_body_name="Sun",
            reference_frame_name=global_frame_orientation,
            aberration_corrections="NONE",
            ephemeris_time=departure_epoch + tof)

        # Create Lambert targeter
        lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
            initial_state[:3], final_state[:3], tof, central_body_gravitational_parameter)

        # Compute initial Cartesian state of Lambert arc
        lambert_arc_initial_state = initial_state
        lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

        # Compute final Cartesian state of Lambert arc
        lambert_arc_final_state = final_state
        lambert_arc_final_state[3:] = lambertTargeter.get_arrival_velocity()

        return lambert_arc_initial_state, lambert_arc_final_state

    def get_excess_vel_vec(self, sc_state, epoch, body):

        # Fetch velocity of the target body
        body_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name=body,
            observer_body_name="Sun",
            reference_frame_name='ECLIPJ2000',
            aberration_corrections="NONE",
            ephemeris_time=epoch)
        body_vel = body_state[3:]
        # Get velocity of the spacecraft
        sc_vel = sc_state[3:]
        # Get relative velocity at arrival - aka excess velocity
        excess_vel_vec = sc_vel - body_vel

        return excess_vel_vec
    
    def get_deltav_capture(self, arrival_excess_vel):

        mu = gravitational_param_dict[self.target_body]  # Define relevant gravitational parameter
        rp = self.sma_tar * (1 - self.ecc_tar)  # Find periapsis of target orbit
        v_target = np.sqrt(mu * (2 / rp - 1 / self.sma_tar))  # Find velocity that needs to be achieved
        v_actual = np.sqrt(arrival_excess_vel**2 + 2 * mu / rp)  # Find actual arrival velocity at periapsis
        deltav_capture = v_actual - v_target

        return deltav_capture

    def get_transfer_parameters(self, departure_epoch, tof):

        # Solve Lambert problem
        initial_lambert_state, final_lambert_state = self.get_lambert_problem_result(departure_epoch, tof)
        # Get departure excess velocity
        departure_excess_vel_vec = self.get_excess_vel_vec(initial_lambert_state, departure_epoch, self.departure_body)
        # Get arrival excess velocity
        arrival_epoch = departure_epoch + tof
        arrival_excess_vel_vec = self.get_excess_vel_vec(final_lambert_state, arrival_epoch, self.target_body)
        # Get capture DeltaV
        arrival_excess_vel_norm = np.linalg.norm(arrival_excess_vel_vec)
        deltav_capture = self.get_deltav_capture(arrival_excess_vel_norm)

        return departure_excess_vel_vec, deltav_capture
    
    def get_moon_departure_deltav(self, excess_vel_vec, sma_dep, ecc_dep, departure_epoch):

        mu_earth = gravitational_param_dict['Earth']
        mu_moon = gravitational_param_dict['Moon']

        # Get magnitude and direction of excess velocity at the boundary of the Earth SoI
        excess_vel_norm = np.linalg.norm(excess_vel_vec)
        excess_vel_versor = excess_vel_vec / excess_vel_norm
        # Get Earth-Moon relative position and state
        moon_state = spice.get_body_cartesian_state_at_epoch(
                target_body_name='Moon',
                observer_body_name="Earth",
                reference_frame_name='ECLIPJ2000',
                aberration_corrections="NONE",
                ephemeris_time=departure_epoch)
        moon_earth_distance = np.linalg.norm(moon_state[:3])
        moon_vel = moon_state[3:]
        # Get magnitude of velocity vector when leaving the Moon SoI
        vel_norm_moon_soi = np.sqrt(excess_vel_norm**2 + 2 * mu_earth / moon_earth_distance)  # Hyperbolic velocity magnitude at Moon's SoI in Earth frame
        vel_vec_moon_soi = vel_norm_moon_soi * excess_vel_versor  # Hyperbolic velocity magnitude at Moon's SoI in Earth frame
        # Get Moon-centered excess velocity
        excess_vel_moon = np.linalg.norm(vel_vec_moon_soi - moon_vel)  # Norm of the excess velocity in the Moon reference frame
        if debug_flag:
            print('Moon velocity:', np.linalg.norm(moon_vel))
            print('Arrival excess velocity at Moon SoI:', excess_vel_moon)

        # Parameters of the parking Moon orbit
        rp = sma_dep * (1 - ecc_dep)  # Periselene radius
        vp_0 = np.sqrt(mu_moon * (2 / rp - 1 / sma_dep))  # Initial velocity at periselene
        vp_needed = np.sqrt(excess_vel_moon**2 + 2 * mu_moon / rp)  # Needed velocity at periselene for departure
        deltav = vp_needed - vp_0

        return deltav

    def get_earth_departure_deltav(self, excess_vel_vec, sma_dep, ecc_dep):

        mu_earth = gravitational_param_dict['Earth']
        excess_vel_norm = np.linalg.norm(excess_vel_vec)
        # Parameters of the parking Moon orbit
        rp = sma_dep * (1 - ecc_dep)  # Pericenter radius
        vp_0 = np.sqrt(mu_earth * (2 / rp - 1 / sma_dep))  # Initial velocity at pericenter
        vp_needed = np.sqrt(excess_vel_norm**2 + 2 * mu_earth / rp)  # Needed velocity at pericenter for departure
        deltav = vp_needed - vp_0

        return deltav
    
    def get_deltav_earth_launch(self, departure_epoch, tof, sma_dep, ecc_dep):
        
        # Get parameters of interplanetary transfer and capture
        departure_excess_vel_vec, deltav_capture = self.get_transfer_parameters(departure_epoch, tof)
        # Get DeltaV needed for departure
        deltav_departure = self.get_earth_departure_deltav(departure_excess_vel_vec, sma_dep, ecc_dep)
        # Get total DeltaV
        deltav_total = deltav_departure + deltav_capture

        return deltav_total
    
    def get_deltav_moon_launch(self, departure_epoch, tof, sma_dep, ecc_dep):
        
        # Get parameters of interplanetary transfer and capture
        departure_excess_vel_vec, deltav_capture = self.get_transfer_parameters(departure_epoch, tof)
        # Get DeltaV needed for departure
        deltav_departure = self.get_moon_departure_deltav(departure_excess_vel_vec, sma_dep, ecc_dep, departure_epoch)
        # Get total DeltaV
        deltav_total = deltav_departure + deltav_capture

        return deltav_total


###########################################################################
# CUSTOM FUNCTIONS ########################################################
###########################################################################

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

    R_moon = average_radius_dict['Moon']
    mu_moon = gravitational_param_dict['Moon']
    p = (R_moon * V0 * np.cos(gamma)) ** 2 / mu_moon
    e = np.sqrt(1 - R_moon * V0 ** 2 / mu_moon * (2 - R_moon * V0 ** 2 / mu_moon) * np.cos(gamma) ** 2)

    ra = p / (1 - e)
    ra_target = R_moon + h
    return ra - ra_target

def initial_launch_angle(gamma: float, V0: float = 1754, h0: float = 100e3):
    return initial_velocity(V0, gamma, h0)


###########################################################################
# TESTING #################################################################
###########################################################################

if __name__ == '__main__':
    # Launch trajectory test
    launch_trajectory = LaunchTrajectory('Moon', 1800, np.radians(10))
    vel = launch_trajectory.get_initial_velocity_for_altitude(100e3)
    launch_trajectory.reinitialize_paramters(np.radians(10), vel)

    print('Initial velocity:', vel)
    print('Max altitude:', round(launch_trajectory.get_max_altitude() / 1e3), ' km')
    print('Delta-V: ', round(launch_trajectory.get_deltav_for_circularization(), 2), ' m/s')

    # Launch trajectory optimization

    # Transfer test
    # parking = Orbit('Earth', 300e3 + average_radius_dict['Earth'], 0)
    # target = Orbit('Mars', 300e3 + average_radius_dict['Mars'], 0)
    # target_2 = Orbit('Moon', 100e3 + average_radius_dict['Moon'], 0)
    #
    # transfer_1 = Transfer(parking, target)
    # transfer_2 = Transfer(target, parking)
    # print(transfer_1.get_transfer_delta_V(True))
    # print(transfer_2.get_transfer_delta_V(True))
    #
    # transfer_3 = Transfer(parking, target_2, 'Earth')
    # transfer_4 = Transfer(target_2, parking, 'Earth')
    # print(transfer_3.get_transfer_delta_V(True))
    # print(transfer_4.get_transfer_delta_V(True))
