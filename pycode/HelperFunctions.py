'''
Helper functions and classes that may be useful when studying patched conics problems. Description TBD
'''

# General imports
import numpy as np

# Tudat imports
from tudatpy.interface import spice
from tudatpy.astro.element_conversion import cartesian_to_keplerian


###########################################################################
# LOAD USEFUL CELESTIAL PARAMETERS ########################################
###########################################################################

# Load spice kernels
spice.load_standard_kernels()

# Gravitational parameters
mu_sun = spice.get_body_gravitational_parameter('Sun')
mu_moon = spice.get_body_gravitational_parameter('Moon')
mu_earth = spice.get_body_gravitational_parameter('Earth')
gravitational_param_dict = {
    'Sun' : mu_sun,
    'Earth' : mu_earth,
    'Moon' : mu_moon,
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


###########################################################################
# CUSTOM CLASSES ##########################################################
###########################################################################

class Orbit:
    '''
    DESCRIPTIONM TBD
    '''
    def __init__(self,
                 orbited_body : str,
                 sma : float,
                 ecc : float = 0,
                 incl : float = np.pi / 2,
                 omega : float = 0,
                 raan : float = 0,
                 theta : float = 0):
        
        # Attribute Keplerian orbital parameters
        self.orbited_body = orbited_body
        self.sma = sma
        self.ecc = ecc
        self.incl = incl
        self.omega = omega
        self.raan = raan
        self.theta = theta

        # Attribute relevant physical environment parameters
        self.mu = gravitational_param_dict[self.orbited_body]

        # Useful orbit variables
        self.rp = self.sma * (1 - self.ecc)  # Get radius of periapsis for ideal impulsive thrust

    def get_velocity_for_escape(self, escape_velocity : float = 0):
        '''
        Description
        '''

        rp = self.sma * (1 - self.ecc)  # Get radius of periapsis for ideal impulsive thrust
        velocity_at_periapsis = np.sqrt(escape_velocity ** 2 + 2 * self.mu / self.rp)

        return velocity_at_periapsis
    
    def get_deltav_for_escape(self, escape_velocity : float = 0):

        # Define nominal velocity at periapsis
        velocity_at_periapsis = np.sqrt(self.mu * (2 / self.rp - 1 / self.sma))
        # Required velocity at periapsis for escape
        required_velocity_at_periapsis = self.get_velocity_for_escape(self, escape_velocity)
        # Needed DeltaV
        deltav = required_velocity_at_periapsis - velocity_at_periapsis

        return deltav



    

        










