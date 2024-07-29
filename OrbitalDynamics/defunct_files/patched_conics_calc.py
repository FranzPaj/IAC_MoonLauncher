'''
Description TBD
'''

# Generic imports
import numpy as np

# Helper Functions import
from pycode import HelperFunctions as Util
from pycode.HelperFunctions import sma_moon, mu_moon


###########################################################################
# ESCAPE VELOCITIES #######################################################
###########################################################################

# Comparison between the DeltaV needed to escape Earth's SoI from a Moon  and from 
# an Earth parking orbit. 
# NOTE Class and structure is definitely still WIP

# Define problem parameters
parking_orbit_altitude_earth = 300 * 10 ** 3  # m
parking_orbit_altitude_moon = 100 * 10 ** 3  # m
required_escape_velocity = 0.0 * 10 ** 3  # m/s

# Define orbits
# Earth-centric orbit
sma_earth_orbit = Util.get_sma_from_altitude('Earth', parking_orbit_altitude_earth)
earth_centric_orbit = Util.Orbit('Earth', sma_earth_orbit)
# Moon-centric orbit
sma_moon_orbit = Util.get_sma_from_altitude('Moon', parking_orbit_altitude_moon)
moon_centric_orbit = Util.Orbit('Moon', sma_moon_orbit)
# Moon orbit
moon_orbit = Util.Orbit('Earth', sma_moon)


# Get deltav from Earth
earth_deltav = earth_centric_orbit.get_deltav_for_escape(required_escape_velocity)
# Get deltav from Moon
excess_velocity_at_moon_soi = moon_orbit.get_deltav_for_escape(required_escape_velocity)
moon_deltav = moon_centric_orbit.get_deltav_for_escape(excess_velocity_at_moon_soi)

print('Earth DeltaV:', earth_deltav)
print('Moon DeltaV:', moon_deltav)







