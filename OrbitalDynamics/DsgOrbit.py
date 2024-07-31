"""
Functions to extracxt a CR3BP periodic orbit for the Deep Space Gateway from the NASA SPICE data. The code first fetches the relevant
ephemerides in the ECLIPJ2000 inertial frame and then converts a state to the Earth-Moon CR3BP. With a typical continuation method the 
complex NASA SPICE orbit initial conditions are converted to a simpler CR3BP periodic Halo orbit, and the starting conditions for this orbit
are then saved for further analysis.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Tudat imports
from tudatpy.interface import spice
from tudatpy import constants

# Custom imports
from pycode import LagrangeUtilities as LagrUtil
from pycode.CustomConstants import mu_combined, MU_ADIMENSIONAL
from pycode import PlotGenerator as PlotGen


###########################################################################
# DEFINE GLOBAL QUANTITIES ################################################
###########################################################################

# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel(os.path.join(current_dir, 'input_data', 'receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp'))


###########################################################################
# FETCH RELEVANT TABULATED DATA ###########################################
###########################################################################

# Find a moment around 25 years after J2000 for which the DSG is at its apex
dsg_period = 28 * 2 / 9 * constants.JULIAN_DAY
t0 = 25 * constants.JULIAN_YEAR  # Epoch - s

num_points = 1000
epoch_ls = np.linspace(t0, t0 + dsg_period, num=num_points)
z_ls = np.zeros(num_points)

for i, epoch in enumerate(epoch_ls):
    # Get initial states
    # Define initial state for Deep Space Gateway 25 years after J2000
    dsg_initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name='-60000',
        observer_body_name='EARTH MOON BARYCENTER',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='NONE',
        ephemeris_time = epoch)
    dsg_initial_pos = dsg_initial_state[:3]
    dsg_initial_vel = dsg_initial_state[3:]

    z_ls[i] = dsg_initial_pos[2]

# Find maximum
maximum_index = np.argmax(z_ls)
# Get definitive starting epoch
epoch = epoch_ls[maximum_index]

# Define initial state for Deep Space Gateway at epoch
dsg_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='-60000',
    observer_body_name='EARTH MOON BARYCENTER',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = epoch)
dsg_initial_pos = dsg_initial_state[:3]
dsg_initial_vel = dsg_initial_state[3:]
# Get initial state at epoch
moon_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='Moon',
    observer_body_name='EARTH MOON BARYCENTER',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = epoch)
moon_initial_pos = moon_initial_state[:3]
moon_initial_vel = moon_initial_state[3:]
# Get initial state at epoch
earth_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='Earth',
    observer_body_name='EARTH MOON BARYCENTER',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = epoch)


###########################################################################
# CONVERT TO CR3BP ########################################################
###########################################################################

#### Get the adimensionalisation factors
# Get length adimensionalisation factor
length_adim_factor = np.linalg.norm(moon_initial_state[:3] - earth_initial_state[:3])  # Earth-Moon distance - m
# Get time adimensionalisation factor
sma_earth_moon = length_adim_factor  # In first approximation, circular motion
t_period = 2 * np.pi * np.sqrt(sma_earth_moon**3 / mu_combined)
mean_motion = 2 * np.pi / t_period
time_adim_factor = 1 / mean_motion

#### Define the new co-rotating frame at t0 and the relative versors
ihat = moon_initial_pos / np.linalg.norm(moon_initial_pos)  # x-axis
# Define orbital plane
L_vec = np.cross(moon_initial_pos, moon_initial_vel)  # Moon angular momentum
khat = L_vec / np.linalg.norm(L_vec)  # z-axis
jhat = np.cross(khat, ihat)

#### Convert the DSG state to CR3BP
new_pos = np.array([np.dot(dsg_initial_pos, ihat), np.dot(dsg_initial_pos, jhat), np.dot(dsg_initial_pos, khat)])  # Convert position
angular_vel_vec = mean_motion * khat  # Define rotating system angular velocity
dsg_initial_vel = dsg_initial_vel - np.cross(angular_vel_vec, dsg_initial_pos)  # Correct for system rotation
new_vel = np.array([np.dot(dsg_initial_vel, ihat), np.dot(dsg_initial_vel, jhat), np.dot(dsg_initial_vel, khat)])  # Convert velocity
new_pos = new_pos / length_adim_factor  # Adimensionalise position
new_vel = new_vel / length_adim_factor * time_adim_factor  # Adimensionalise verlocity
new_state = np.concatenate((new_pos, new_vel))
dsg_x0_nasa = new_state  # CR3BP DSG initial state obtained


###########################################################################
# OBTAIN CLOSED ORBIT #####################################################
###########################################################################

# Define parameters for integration
T_nom = 2 * np.pi * 2 / 9
T_period = 29.53049 / 27.32158 * T_nom  # Conversion from sydereal period to synodic period
T_period = 30.5 / 27.32158 * T_nom  # Super-Manual and ugly correction because continuation is not super-precise
time_span_orbit = [0, T_period]  # Span of a typical DSG orbit
tol_orbit = 1.0E-12  # Relative and absolute tolerance for integration
tol_correction = 1.0e-13  # Absolute tolerance for error in orbit continuation
args = (MU_ADIMENSIONAL, 0, 'constant', np.array([0,0,0]))


# Get the first, uncorrected orbit
t, y_nasa = LagrUtil.propagate_3d_orbit(dsg_x0_nasa, time_span_orbit, tol_orbit, args)

# Get a corrected starting condition with the continuation method
x0_corrected, iter = LagrUtil.orbit_continuation(dsg_x0_nasa, MU_ADIMENSIONAL, print_flag=False)
print('Needed correction:', x0_corrected - dsg_x0_nasa)
# Get the corrected orbit
t, y_corrected = LagrUtil.propagate_3d_orbit(x0_corrected, time_span_orbit, tol_correction, args)

# Define cases to display in graphs
cases_dict = {
    'nasa':{
        'states':y_nasa,
        'color':'black',
        'linestyle':'-',
        'label':'NASA SPICE'
    },
    'corrected':{
        'states':y_corrected,
        'color':'blue',
        'linestyle':'-',
        'label':'corrected for CR3BP'
    },
}
cases_displayed = ['nasa', 'corrected']

# Obtain graphs
fig = PlotGen.halo_plot(cases_dict, cases_displayed)
plt.show()
















