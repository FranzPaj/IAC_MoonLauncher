'''
File to define and/or fetch useful cosmic constants.
'''

# General imports
import numpy as np

# Tudat imports
from tudatpy.interface import spice
from tudatpy import constants

spice.load_standard_kernels()


###########################################################################
# DEFINE THE CONSTANTS ####################################################
###########################################################################

# Fetch gravitational parameters from SPICE
# Gravitational parameters
mu_sun = spice.get_body_gravitational_parameter('Sun')
mu_moon = spice.get_body_gravitational_parameter('Moon')
mu_earth = spice.get_body_gravitational_parameter('Earth')
mu_mars = spice.get_body_gravitational_parameter('Mars')
gravitational_param_dict = {
    'Sun': mu_sun,
    'Earth': mu_earth,
    'Moon': mu_moon,
    'Mars': mu_mars,
}
# Average radius
R_sun = spice.get_average_radius('Sun')
R_moon = spice.get_average_radius('Moon')
R_earth = spice.get_average_radius('Earth')
R_mars = spice.get_average_radius('Mars')
average_radius_dict = {
    'Sun': R_sun,
    'Earth': R_earth,
    'Moon': R_moon,
    'Mars': R_mars,
}
# Average SMA
sma_dict = {
    'Mercury': 57.9e9,
    'Venus': 108.2e9,
    'Earth': 149.6e9,
    'Mars': 227.9e9,
    'Jupiter': 778.6e9,
    'Saturn': 1433.5e9,
    'Uranus': 2872.5e9,
    'Neptune': 4495.1e9,
    'Moon': 384.4e6,
}
# Get Earth-Moon combined gravitational parameter
mu_combined = gravitational_param_dict['Moon'] + gravitational_param_dict['Earth']


#### Constants for CR3BP
EM_DISTANCE = 382606991.2018355  # Earth-Moon distance, replace with better formulation - m
R_M = spice.get_average_radius('Moon')  # Average Moon radius
MU_ADIMENSIONAL = mu_moon / mu_combined  # Gravitational parameter for adimensionalised CR3BP
R_M_ADIMENSIONAL = R_M / EM_DISTANCE






