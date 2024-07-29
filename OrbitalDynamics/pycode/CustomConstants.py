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
mu_earth = spice.get_body_gravitational_parameter('Earth')
mu_moon = spice.get_body_gravitational_parameter('Moon')
mu_combined = mu_earth + mu_moon
EM_DISTANCE = 382606991.2018355  # Earth-Moon distance, replace with better formulation - m
R_M = spice.get_average_radius('Moon')  # Average Moon radius

#### Constants for CR3BP
MU_ADIMENSIONAL = mu_moon / mu_combined  # Gravitational parameter for adimensionalised CR3BP
R_M_ADIMENSIONAL = R_M / EM_DISTANCE




