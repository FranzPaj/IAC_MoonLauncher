import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from pycode.HelperFunctions import gravitational_param_dict, average_radius_dict
import warnings

###########################################################################
# CIRCULAR MOON ###########################################################
###########################################################################

'''
Functions for modeling Moon launch, assuming a planar, circular Moon.
Still requires some checking and validation (who doesn't, ammiright?).
'''

def polar_int_fun(t, state, thrust, g, Isp):
    '''
    Integration helper function for the IVP solver. Takes as an input the state and returns the state 
    derivatives with regard to time.
    '''

    r = state[0]  # Radial distance from Moon
    theta = state[1]  # Polar angle
    gamma = state[2]  # Flight-path angle
    v = state[3]  # Planar velocity
    mtot = state[4]  # Total mass
    mp = state[5]  # Mass of propellant to be burnt

    # Check if propellant is finished
    if mp <= 0:
        thrust = 0

    rdot = v * np.sin(gamma)  # From simple kinematics
    thetadot = v * np.cos(gamma) / r  # From simple kinematics
    gammadot = - 1 / v * g * np.cos(gamma) * (1 - v**2 / (g * r))  # From Re-entry Systems - I'll share the sauce later, Javi
    vdot = thrust / mtot - g * np.sin(gamma)  # From Rocket Motion (and simple dynamics)
    mdot = - thrust / (Isp * 9.81)  # Better than Tsiolkovskji, does not require integration with time

    state_dot = np.array([rdot, thetadot, gammadot, vdot, mdot, mdot])

    return state_dot

def get_tangency_condition(gamma0, v0, thrust, g, Isp, construction_ratio, m0, ratio_guess):
    '''
    Function for finding the altitude over the Moon for which tangency of a constant thrust gravity turn occurs for
    a given propellant ratio.
    '''

    mtot0 = m0  # Total mass at t0
    mp0 = m0 * (1 - construction_ratio) * ratio_guess  # Propellant to be burnt during ascent
    r0 = average_radius_dict['Moon'] + 1e-6  # Radius at t0
    theta0 = 0  # Polar angle at t0

    # Integrate EoMs
    # Define stop condition when rocket reaches apogee
    def stop_condition(t, state, thrust, g, Isp):
        return state[2]
    stop_condition.terminal = True
    # Define arguments
    args = (thrust, g, Isp)
    # Define parameters for integration
    time_span_orbit = [0, 24 * 3600]  # Dummy time span of 1 day, integration will be done earlier though
    tol = 1.0e-12 # Relative and absolute tolerance for integration
    # Define initial state
    x0 = np.array([r0, theta0, gamma0, v0, mtot0, mp0])
    # Execute integration
    sol = scipy.integrate.solve_ivp(polar_int_fun, time_span_orbit, x0, method='RK45',
                                rtol=tol, atol=tol, args=args, events=stop_condition)
    y = sol.y.T  # Get history of state vectors
    h_fin = y[-1,0] - average_radius_dict['Moon']  # Get tangency altitude over Moon surface

    return h_fin

def get_orbit_reaching_condition(gamma0, v0, thrust, g, Isp, construction_ratio, m0, prop_ratio, h_th):
    '''
    Function for finding the state with which the spacecraft reaches a given altitude. Useful for getting
    needed quantities for the final propellant ratio.
    '''

    mtot0 = m0  # Total mass at t0
    mp0 = m0 * (1 - construction_ratio) * prop_ratio  # Propellant to be burnt during ascent
    r0 = average_radius_dict['Moon'] + 1e-6  # Radius at t0 - m
    theta0 = 0  # Polar angle at t0

    # Integrate EoMs
    # Define stop condition when rocket reaches apogee
    def stop_condition(t, state, thrust, g, Isp):
        return state[0] - (average_radius_dict['Moon'] + h_th - 6e3)

    stop_condition.terminal = True

    # Define arguments - constants useful during the integration
    args = (thrust, g, Isp)
    # Define parameters for integration
    time_span_orbit = [0, 24 * 3600]
    tol = 1.0e-12 # Relative and absolute tolerance for integration
    # Define initial state
    x0 = np.array([r0, theta0, gamma0, v0, mtot0, mp0])
    sol = scipy.integrate.solve_ivp(polar_int_fun, time_span_orbit, x0, method='RK45',
                                    rtol=tol, atol=tol, args=args, events=stop_condition)

    y = sol.y.T  # Get history of state vectors
    h_fin = y[-1,0] - average_radius_dict['Moon']  # Get tangency altitude over Moon surface
    mtot_fin = y[-1, 4]  # Get final mass
    v_fin = y[-1, 3]  # Get final velocity
    gamma_fin = y[-1,2]  # Get final flight-path anlge
    mp_fin = y[-1,5]  # Get final propellant-to-be-burnt mass

    return h_fin, mtot_fin, v_fin, gamma_fin, mp_fin

def launch_circular(gamma0, v0,
           Isp: float = 300,
           construction_ratio: float = 0.1,
           mass: float = 1000,
           thrust: float = 1000):
    '''
    Function for finding the propellant ratio for which tangency at a given altitude occurs,using a constant thrust gravity turn.
    The solution-searching algorithm searches the ratio interval [0, 1] by bisecting the interval at each passage and verifying
    whether the solution is in the lower or higher half. This assumes that the function is continuous and monotone, which seems 
    about right honestly.
    '''

    g = 1.625  # Gravity on the Moon
    v_target = np.sqrt(gravitational_param_dict['Moon'] / (average_radius_dict['Moon'] + 1e5))  # Orbital velocity at 100 km

    h_th = 1e5  # Define threshold altitude
    convergence_flag = False  # Define flag used to check whether solution has been found
    convergence_range = 1e-2  # Define precision requested for finding the correct ratio (given in absolute value)
    unfeasible_flag = False  # Define flag to check whether a gravity turn is achievable for the given initial conditions

    # Start bisection algorithm to find tangency condition - the ratio search space is [0, 1]
    ratio_low_guess = 0
    ratio_high_guess = 1

    # Evaluate the tangency altitude for the extrema of the search space
    h_low_guess = get_tangency_condition(gamma0, v0, thrust, g, Isp, construction_ratio, mass, ratio_low_guess)
    h_high_guess = get_tangency_condition(gamma0, v0, thrust, g, Isp, construction_ratio, mass, ratio_high_guess)

    # Check if no propellant is already sufficient
    if h_low_guess > h_th:
        ratio_sol = ratio_low_guess
        convergence_flag = True
    # Check if trajectory is unfeasible even burning all the propellant
    if h_high_guess < h_th:
        ratio_sol = np.nan
        convergence_flag = True
        unfeasible_flag = True
    # For all other cases, the solution exists! :)

    # Initialise middle point
    ratio_middle_guess = (ratio_high_guess + ratio_low_guess) / 2

    while not convergence_flag:

        # Evaluate tangency at middle point
        h_middle_guess = get_tangency_condition(gamma0, v0, thrust, g, Isp, construction_ratio, mass, ratio_middle_guess)

        # Check whether the middle point will be the new high or low extreme of the interval
        if h_middle_guess > h_th:
            ratio_high_guess = ratio_middle_guess
        else:
            ratio_low_guess = ratio_middle_guess

        # Find new middle point
        ratio_middle_guess = (ratio_high_guess + ratio_low_guess) / 2

        # Check whether we reached convergence
        if (ratio_high_guess - ratio_low_guess) / 2 < convergence_range:
            convergence_flag = True
            ratio_sol = (ratio_high_guess + ratio_low_guess)/2

    # If gravity turn is feasible, calculate the gear ratio when accounting for the circularisation manoeuvre
    if not unfeasible_flag:

        # Calculate final impulse for circularization
        h_fin, mtot_fin, v_fin, gamma_fin, mp_fin = get_orbit_reaching_condition(gamma0, v0, thrust, g, Isp, construction_ratio, mass, ratio_sol, h_th)
        if h_fin < 0:
            ratio_sol = np.nan
        deltaV = np.sqrt(v_fin ** 2 + v_target ** 2 - 2 * v_fin * v_target * np.cos(gamma_fin))  # From cosine law
        dm_manoeuvre = mtot_fin * (1 - np.exp(-deltaV / (Isp * 9.81)))  # Defined as > 0

        ### Calculate gear ratio (ratio between the total propellant at t0 and the propellant that can be delivered) ###
        mp_initial = mass * (1 - construction_ratio)  # Initial propellant
        mp_pl = mass * (1 - construction_ratio) * (1 - ratio_sol)  # Payload propellant
        mp_leftover = mp_fin  # Leftover propellant (if not all assigned prop was burnt during ascent)
        mp_end = mp_pl + mp_leftover - dm_manoeuvre  # Propellant in orbit

        gr = mp_initial / mp_end  # Gear ratio

    else:
        gr = np.nan
    
    return gr


##### TODO Launch_alt function, to be removed after verification of the new launch_circular
def launch_alt(angle, v0,
           Isp: float = 300,
           construction_ratio: float = 0.1,
           mass: float = 1000,
           thrust: float = 1000):
    '''Equations of motion for gravity turn in polar coordinates'''

    g = 1.625  # Gravity on the Moon
    v_target = np.sqrt(gravitational_param_dict['Moon'] / (average_radius_dict['Moon'] + 1e5))  # Orbital velocity

    dt = 1e-2
    prop_mass_ratio = 1 / 10

    convergence = False
    falling = False
    removed_mass = False

    while not convergence:
        total_mass = mass
        mp = total_mass * (1 - construction_ratio) * prop_mass_ratio  # Propellant to be burnt during ascent
        dry_mass = total_mass - mp

        t = 0
        v = v0
        gamma = angle
        pos = np.array([[average_radius_dict['Moon'], 0]])
        new_r = average_radius_dict['Moon']

        # Start simulation
        while gamma >= 0:

            current_mass = dry_mass + mp
            current_r = new_r

            # Calculate thrust
            if mp <= 0:
                thrust = 0

            # EOM
            dVdt = thrust / current_mass - g * np.sin(gamma)
            dGdt = 1 / v * (-g) * np.cos(gamma) * (1 - v**2 / (g * current_r) )

            # Update the state
            gamma += dGdt * dt
            v += dVdt * dt
            t += dt

            # Update mass
            dm = current_mass * (1 - np.exp(-dVdt * dt / (Isp * 9.81)))
            current_mass -= dm
            mp -= dm

            # Store position values and check end conditions
            dr = v * np.sin(gamma) * dt
            dtheta = v * np.cos(gamma) / current_r * dt
            pos = np.vstack((pos, pos[-1] + np.array([dr, dtheta])))

            if pos[-1, 0] < pos[-2, 0]:  # We are falling
                falling = True
                break

            if pos[-1, 0] > average_radius_dict['Moon'] + 1e5:  # We are overshooting
                break

            new_r = pos[-1, 0]

        # Analyze exit conditions
        if gamma <= np.radians(5) and pos[-1, 0] >= average_radius_dict['Moon'] + 1e5:  # Solution found
            convergence = True

        if falling:
            if removed_mass:
                # The solution lies between the previous and current mass fractions
                prop_mass_ratio += 1 / 20
                convergence = True
            else:
                prop_mass_ratio += 1 / 20  # Add propellant mass
                if prop_mass_ratio >= 1:
                    # Not enough mass
                    return np.nan

        else:
            if pos[-1, 0] > average_radius_dict['Moon'] + 1e5:  # Overshoot
                # Rocket could have burnt less mass
                prop_mass_ratio -= 1 / 20
                removed_mass = True
                if prop_mass_ratio <= 0:
                    # Solution lies between 0 and 1/20 propellant mass ratio
                    prop_mass_ratio += 1 / 20
                    convergence = True
            else:  # Undershoot
                prop_mass_ratio += 1 / 20
                if removed_mass:
                    # The solution lies between the previous and current mass fractions
                    convergence = True

    # Calculate final impulse for circularization
    deltaV = np.sqrt(v ** 2 + v_target ** 2 - 2 * v * v_target * np.cos(gamma))  # From cosine law
    dm_manoeuvre = current_mass * (1 - np.exp(-deltaV / (Isp * 9.81)))  # Defined as > 0

    ### Calculate gear ratio (ratio between the total propellant at t0 and the propellant that can be delivered) ###
    mp_initial = total_mass * (1 - construction_ratio)
    mp_final = total_mass * (1 - construction_ratio) * (1 - prop_mass_ratio) - dm_manoeuvre
    gr = mp_initial / mp_final  # Gear ratio

    # plt.plot(pos[:,0] * np.cos(pos[:,1]), pos[:,0] * np.sin(pos[:,1]))
    # plt.show()
    # plt.plot(np.arange(0,len(pos[:,0])) * 0.01, (pos[:,0] - average_radius_dict['Moon']) / 1000)
    # plt.show()

    return gr


def optimize_initial_params(gamma: np.ndarray, v0: np.ndarray):
    # Initialize arrays to save solution
    useful_mass_fraction = np.zeros((len(gamma), len(v0)))

    # Start simulation
    for i in tqdm(range(len(gamma))):
        angle = gamma[i]
        for j, vel in enumerate(v0):
            useful_mass_fraction[i, j] = launch_circular(angle, vel)

    return useful_mass_fraction


if __name__ == '__main__':
    # Design range
    gamma = np.radians(np.arange(0, 90, 5))
    v0 = np.arange(200, 2000, 100)

    useful_mass_fraction = optimize_initial_params(gamma, v0)

    for i in range(len(useful_mass_fraction[:,0])):
        row = useful_mass_fraction[i,:]
        if np.isnan(row).all():
            continue

        velocity = np.where(row == np.nanmin(row))[0][0]

        plt.scatter(v0[velocity], gamma[i], color='r')  # 'ro' plots a red dot

    # Continue with the existing plotting code
    plt.imshow(useful_mass_fraction, vmin=1, vmax=3)

    # Plot color grid
    gr_min = 1
    gr_max = 4
    # Define color scale
    cbar = plt.colorbar()
    cbar.set_label('Gear ratio')
    plt.xticks(ticks=np.arange(0, len(v0), 2), labels=np.array(np.round(v0[::2]), dtype=int))
    plt.yticks(ticks=np.arange(0, len(gamma), 5), labels=np.array(np.round(np.degrees(gamma[::5])), dtype=int))

    plt.xlabel('Initial velocity [m/s]')
    plt.ylabel('Launch angle [degrees]')
    plt.title('Mass = 1T, Isp = 300s, Mc/M = 0.1, Thrust = 1kN')

    plt.show()

