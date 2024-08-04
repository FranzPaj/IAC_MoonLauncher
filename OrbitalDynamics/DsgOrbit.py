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
import scipy

# Tudat imports
from tudatpy.interface import spice
from tudatpy import constants

# Custom imports
from pycode import LagrangeUtilities as LagrUtil
from pycode import HelperFunctions as Util
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

plots_dir = os.path.join(current_dir, 'plots', 'dsg')

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
T_period_manual = 29.53049 / 27.32158 * T_nom  # Conversion from sydereal period to synodic period
T_period_manual = 30.5 / 27.32158 * T_nom  # Super-Manual and ugly correction because continuation is not super-precise
tol = 1.0E-12  # Relative and absolute tolerance for integration
tol_correction = 1.0e-13  # Absolute tolerance for error in orbit continuation
args = (MU_ADIMENSIONAL, 0, 'constant', np.array([0,0,0]))

# Get x vector when xz plane is crossed
# Define stop condition after one orbit
def stop_condition(t, state, mu, beta, orientation_type, orientation):
    return state[1]
stop_condition.terminal = True
time_span_orbit_reverse = [10 * np.pi, 0]  # Propagate backwards
sol = scipy.integrate.solve_ivp(LagrUtil.fun_3d, time_span_orbit_reverse, dsg_x0_nasa, method='RK45',
                                rtol=tol, atol=tol, args=args, events=stop_condition)
x0_approx = sol.y.T[-1, :6]
# Correct this initial guess
x0_approx[1] = x0_approx[1] + 1e-14  # Avoid stopping at t0
x0_approx[3] = 0  # vx = 0
x0_approx[5] = 0  # vz = 0
print(x0_approx)


# Get the first, uncorrected orbit
# Define stop condition after one orbit
def stop_condition(t, state, mu, beta, orientation_type, orientation):
    return state[1]
stop_condition.terminal = True
stop_condition.direction = 1.0  # Only stop when y becomes positive again
time_span_orbit = [0, 10 * np.pi]
sol = scipy.integrate.solve_ivp(LagrUtil.fun_3d, time_span_orbit_reverse, x0_approx, method='RK45',
                                rtol=tol, atol=tol, args=args, events=stop_condition)
y_nasa = sol.y.T

# Get a corrected starting condition with the continuation method
x0_corrected, iter = LagrUtil.orbit_continuation(x0_approx, MU_ADIMENSIONAL, tol=tol_correction, print_flag=False)

print('Needed correction:', x0_corrected - x0_approx)
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
# plt.show()

print('Orbit successfully corrected!')
print('-------------------------------------------------------------------------')


###########################################################################
# OBTAIN MANIFOLDS ########################################################
###########################################################################

# Assess the stability of the monodromy matrix
# Define stop condition after one orbit
def stop_condition(t, state, mu, beta, orientation_type, orientation):
    return state[1]
stop_condition.terminal = True
stop_condition.direction = 1.0  # Only stop when y becomes positive again
args = (MU_ADIMENSIONAL, 0, 'constant', np.array([1,0,0]))
# Define parameters for integration
time_span_orbit = [0, 10*np.pi]  # Arbitrarily large interval
tol = 1.0E-12 # Relative and absolute tolerance for integration
# Define initial state with State Transition Matrix
x0_corrected_with_stm = np.concatenate((x0_corrected, np.eye(6).flatten()))

# Propagate
# First time to get x0 when y is crossed
sol = scipy.integrate.solve_ivp(LagrUtil.fun_var, time_span_orbit, x0_corrected_with_stm, method='RK45',
                                rtol = tol, atol = tol, args = args, events = stop_condition)
x0 = sol.y.T[-1, :6]
x0[1] = x0[1] + 1e-14  # Avoid stopping at t0
x0_with_stm = np.concatenate((x0, np.eye(6).flatten()))
# Second time to get period
sol = scipy.integrate.solve_ivp(LagrUtil.fun_var, time_span_orbit, x0_with_stm, method='RK45',
                                rtol=tol, atol=tol, args=args, events=stop_condition)
t_period = sol.t[-1]
# Third time to get points
num_orbit_starting_points = 10
t_eval = np.linspace(0, t_period, num_orbit_starting_points + 1)[1:]  # Exclude the starting state
sol = scipy.integrate.solve_ivp(LagrUtil.fun_var, time_span_orbit, x0_with_stm, method='RK45', t_eval=t_eval,
                                rtol=tol, atol=tol, args=args, events=stop_condition)

# Extract results
y = sol.y.T
t = sol.t
states = y[:,:6]
stm_list = np.reshape(y[:,6:],(num_orbit_starting_points,6,6))
monodromy_matrix = stm_list[-1,:,:]

print('Period:', t_period)
# print('Monodromy matrix:', stm_list[-1,:,:])
print('Distance between propagated final point and initial point:', np.linalg.norm(states[-1,:3] - x0[:3]))
# Find and print eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(monodromy_matrix)
eigenvectors = eigenvectors.T  # I prefer vector list in direction 0

for i in range(6):
    print(f'lambda_{i}:',eigenvalues[i])

print('-------------------------------------------------------------------------')

print('Getting the orbit stable and unstable manifold tubes')

normalise_flag = True

# Define the stable and unstable eigenvectors for the monodromy matrix
ind_unstable = 0
ind_stable = 1
monodromy_unstable_eigenvector = np.real(eigenvectors[ind_unstable,:])  # Discard imaginary part, since it is == 0
monodromy_stable_eigenvector = np.real(eigenvectors[ind_stable,:])  # Samesies
if normalise_flag:
    monodromy_unstable_eigenvector = monodromy_unstable_eigenvector / np.linalg.norm(monodromy_unstable_eigenvector)
    monodromy_stable_eigenvector = monodromy_stable_eigenvector / np.linalg.norm(monodromy_stable_eigenvector)


print('Monodromy matrix unstable eigenvector:', monodromy_unstable_eigenvector)
print('Monodromy matrix stable eigenvector:', monodromy_stable_eigenvector)

# Define common integration parameters
t0 = 0
t_int = 2 * np.pi * 5  # Integration time (5 lunar sydereal months, but time is normalised so 1 month = 2pi)

# Define magnitude of the perturbation
epsilon = 1.0E-7

# Store the manifolds in a dictionary
starting_points_dict = dict()

print('--*--')

for i in range(num_orbit_starting_points):

    data_dict = dict()
    unperturbed_initial_state = states[i,:]
    data_dict['unperturbed_initial_state'] = unperturbed_initial_state
    stm = stm_list[i,:,:]
    data_dict['STM'] = stm

    # Get the eigenvectors
    stable_eigenvector = np.matmul(stm, monodromy_stable_eigenvector)
    unstable_eigenvector = np.matmul(stm, monodromy_unstable_eigenvector)
    if normalise_flag:
        stable_eigenvector = stable_eigenvector / np.linalg.norm(stable_eigenvector)
        unstable_eigenvector = unstable_eigenvector / np.linalg.norm(unstable_eigenvector)


    # Define a dictionary with all the cases
    manifold_dict = {
        'stable_positive':{
            'time_span': [t0, - t_int],
            'initial_state': unperturbed_initial_state + stable_eigenvector * epsilon,
            'color':'green',
            'linestyle':'solid',
            'label':r'stable, $\epsilon > 0$'
        },
        'stable_negative':{
            'time_span': [t0, - t_int],
            'initial_state': unperturbed_initial_state - stable_eigenvector * epsilon,
            'color':'green',
            'linestyle':'dashed',
            'label':r'stable, $\epsilon < 0$'
        },
        'unstable_positive':{
            'time_span': [t0, t_int],
            'initial_state': unperturbed_initial_state + unstable_eigenvector * epsilon,
            'color':'red',
            'linestyle':'solid',
            'label':r'unstable, $\epsilon > 0$'
        },
        'unstable_negative':{
            'time_span': [t0, t_int],
            'initial_state': unperturbed_initial_state - unstable_eigenvector * epsilon,
            'color':'red',
            'linestyle':'dashed',
            'label':r'unstable, $\epsilon < 0$'
        },
    }

    # Calculate all the manifolds
    for case in manifold_dict.keys():
        # Get the right variables for the specific manifold
        perturbed_initial_state = manifold_dict[case]['initial_state']
        time_span = manifold_dict[case]['time_span']
        # Perform the propagation
        args = (MU_ADIMENSIONAL, 0, 'constant', np.array([1,0,0]))
        t, y = LagrUtil.propagate_3d_orbit(perturbed_initial_state, time_span, tol, args)
        # Save the results
        manifold_dict[case]['states'] = y
        manifold_dict[case]['times'] = t


    # Save the data
    data_dict['manifolds'] = manifold_dict
    starting_points_dict[i] = data_dict

    print(f'Propagations for starting point {i} completed')
    print('--*--')

print('Finished the propagations')
print('-------------------------------------------------------------------------')

# Plot the manifolds
print('Plotting the manifolds...')

# Case 1: only from initial point
ind_to_plot = 9
# Plot xy
fig_topic = 'dsg/orbit_manifolds/one_manifold_xy'
fig = PlotGen.single_manifold_xy(starting_points_dict[ind_to_plot]['manifolds'])
Util.fig_saver(fig_topic,fig)
# Plot xy - closeup
fig_topic = 'dsg/orbit_manifolds/one_manifold_xy_closeup'
fig = PlotGen.single_manifold_xy_closeup(starting_points_dict[ind_to_plot]['manifolds'])
Util.fig_saver(fig_topic,fig)
# Plot xy - departure
fig_topic = 'dsg/orbit_manifolds/one_manifold_xy_departure'
fig = PlotGen.single_manifold_xy_departure(starting_points_dict[ind_to_plot]['manifolds'])
Util.fig_saver(fig_topic,fig)
# Plot xz
fig_topic = 'dsg/orbit_manifolds/one_manifold_xz'
fig = PlotGen.single_manifold_xz(starting_points_dict[ind_to_plot]['manifolds'])
Util.fig_saver(fig_topic,fig)
# Plot xz - closeup
fig_topic = 'dsg/orbit_manifolds/one_manifold_xz_closeup'
fig = PlotGen.single_manifold_xz_closeup(starting_points_dict[ind_to_plot]['manifolds'])
Util.fig_saver(fig_topic,fig)

# Case 2: all manifolds
# Plot xy
fig_topic = 'dsg/orbit_manifolds/all_manifolds_xy'
fig = PlotGen.all_manifolds_xy(starting_points_dict)
Util.fig_saver(fig_topic,fig)
# Plot xy - closeup
fig_topic = 'dsg/orbit_manifolds/all_manifolds_xy_closeup'
fig = PlotGen.all_manifolds_xy_closeup(starting_points_dict)
Util.fig_saver(fig_topic,fig)
# Plot xz
fig_topic = 'dsg/orbit_manifolds/all_manifold_xz'
fig = PlotGen.all_manifolds_xz(starting_points_dict)
Util.fig_saver(fig_topic,fig)
# Plot xz - closeup
fig_topic = 'dsg/orbit_manifolds/all_manifold_xz_closeup'
fig = PlotGen.all_manifolds_xz_closeup(starting_points_dict)
Util.fig_saver(fig_topic,fig)
# Plot yz
fig_topic = 'dsg/orbit_manifolds/all_manifold_yz'
fig = PlotGen.all_manifolds_yz(starting_points_dict)
Util.fig_saver(fig_topic,fig)
# Plot yz - closeup
fig_topic = 'dsg/orbit_manifolds/all_manifold_yz_closeup'
fig = PlotGen.all_manifolds_yz_closeup(starting_points_dict)
Util.fig_saver(fig_topic,fig)



print('Done!')

print('-------------------------------------------------------------------------')

















