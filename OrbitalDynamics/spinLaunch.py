import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from OrbitalDynamics.pycode.HelperFunctions import gravitational_param_dict, average_radius_dict
import warnings
from scipy.optimize import minimize
from scipy.interpolate import griddata


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
    if type(v0) is np.ndarray:
        v = v0[0]
        x0 = np.array([r0, theta0, gamma0, v, mtot0, mp0])
    else:
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
    if type(v0) is np.ndarray:
        v = v0[0]
        x0 = np.array([r0, theta0, gamma0, v, mtot0, mp0])
    else:
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
           thrust: float = 1000,
           return_mass_ratio: bool = False,
           full_output: bool = False):
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
        if h_fin < 90e3:
            ratio_sol = np.nan

        if np.allclose(h_fin, 100e3):
            deltaV = np.sqrt(v_fin ** 2 + v_target ** 2 - 2 * v_fin * v_target * np.cos(gamma_fin))  # From cosine law
        else:
            # Final altitude is not 100 km, we need to calculate the deltaV for the manoeuvre
            # From conservation of energy:
            mu = gravitational_param_dict['Moon']
            R = average_radius_dict['Moon']
            v = np.sqrt(2 * (v_fin**2 / 2 - mu / (R + h_fin) + mu / (R + 100e3)))  # From energy conservation

            # From eccentricity relationship to velocity, radius and angle:
            a = v_fin**2 * (R + h_fin) / mu
            ecc = np.sqrt(1 - a * (2 - a) * np.cos(gamma_fin)**2)
            angle = np.arccos(np.sqrt((1 - ecc**2) / (v**2 * (R + 100e3) / mu * (2 - v**2 * (R + 100e3) / mu))))

            # Delta-V needed:
            deltaV = np.sqrt(v ** 2 + v_target ** 2 - 2 * v * v_target * np.cos(angle))  # From cosine law
        dm_manoeuvre = mtot_fin * (1 - np.exp(-deltaV / (Isp * 9.81)))  # Defined as > 0

        ### Calculate gear ratio (ratio between the total propellant at t0 and the propellant that can be delivered) ###
        mp_initial = mass * (1 - construction_ratio)  # Initial propellant
        mp_pl = mass * (1 - construction_ratio) * (1 - ratio_sol)  # Payload propellant
        mp_leftover = mp_fin  # Leftover propellant (if not all assigned prop was burnt during ascent)
        mp_end = mp_pl + mp_leftover - dm_manoeuvre  # Propellant in orbit

        if return_mass_ratio:
            if full_output:
                return (mp_initial - mp_end) / (mass * construction_ratio + mp_end), mp_end, dm_manoeuvre, h_fin, gamma_fin, v_fin
            else:
                return (mp_initial - mp_end) / (mass * construction_ratio + mp_end)

        gr = mp_initial / mp_end  # Gear ratio

    else:
        if full_output:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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


def launch_funct(vel, angle):
    ratio = launch_circular(angle, vel, return_mass_ratio=True)
    return ratio


def optimize_initial_params_rough():
    min_v = np.array([1560, 1510, 1480, 1450, 1420, 1400, 1380, 1360, 1340, 1320, 1310, 1290, 1280, 1260, 1250, 1240, 1230,
                      1220, 1210, 1200, 1190, 1180, 1170, 1160, 1150, 1140, 1140, 1130, 1120, 1120, 1110, 1100, 1100, 1090,
                      1080, 1080, 1070, 1070, 1060, 1060, 1050, 1040, 1030, 1010, 1000,  990,  980,  970, 960,  950,  940,
                      920,  910,  900,  890,  880,  870,  860,  850,  840,  830,  820,  820,  810,  800,  790,  780,  770,
                      760,  760,  750,  740,  730,  720,  720,  710,  700,  700,  690,  680,  680,  670,  660,  660,  650,
                      640,  640,  630,  630,  620,  620,  610,  610,  600,  600,  590,  590,  580,  580,  570,  570,  560,
                      560,  550,  550,  540,  540,  540,  530,  530,  520,  520,  520,  510,  510,  510,  500,  500,  490,
                      490,  490,  480,  480,  480,  480,  470,  470,  470,  460,  460,  460,  450,  450,  450,  450,  440,
                      440,  440,  440,  430,  430,  430,  430,  420,  420,  420,  420,  410,  410,  410,  410,  410,  400,
                      400,  400,  400,  390,  390,  390,  390,  390,  380,  380,  380,  380,  380,  370,  370,  370,  370,
                      370,  360,  360,  360,  360,  350,  350,  350,  340])

    opt_v = []
    opt_ratio = []
    gamma = np.radians(np.arange(0.5, 90, 0.5))

    for i in tqdm(range(len(gamma))):
        previous = 1000
        v = min_v[i]

        mass_ratio = launch_circular(gamma[i], v, return_mass_ratio=True)
        while previous > mass_ratio:
            if np.degrees(gamma[i]) < 20:
                v += 0.1
            else:
                v += 1
            previous = mass_ratio
            mass_ratio = launch_circular(gamma[i], v, return_mass_ratio=True)

        opt_v.append(v - 1)
        opt_ratio.append(previous)

    return opt_v, opt_ratio


def fill_nan_2d(array):
    # Get the indices of the valid values
    valid_mask = ~np.isnan(array)
    valid_coords = np.array(np.nonzero(valid_mask)).T
    valid_values = array[valid_mask]

    # Get the indices of the nan values
    nan_coords = np.array(np.nonzero(~valid_mask)).T

    # Interpolate the nan values
    filled_values = griddata(valid_coords, valid_values, nan_coords, method='linear')

    # Fill the nan values in the original array
    filled_array = array.copy()
    filled_array[~valid_mask] = filled_values

    return filled_array


def optimize_mass_ratio(gamma: np.ndarray,
                        v0: np.ndarray,
                        Isp: float = 300,
                        plot_mass_ratio: bool = False,
                        mass_diagnostics: bool = False):
    # Initialize arrays to save solution
    mass_ratio = np.zeros((len(gamma), len(v0)))
    end_propellant = np.zeros((len(gamma), len(v0)))
    circularization_mass = np.zeros((len(gamma), len(v0)))
    final_height = np.zeros((len(gamma), len(v0)))
    final_angle = np.zeros((len(gamma), len(v0)))
    final_vel = np.zeros((len(gamma), len(v0)))

    min_v = np.array(
        [1560, 1510, 1480, 1450, 1420, 1400, 1380, 1360, 1340, 1320, 1310, 1290, 1280, 1260, 1250, 1240, 1230, 1220,
         1210, 1200, 1190, 1180, 1170, 1160, 1150, 1140, 1140, 1130, 1120, 1120, 1110, 1100, 1100, 1090, 1080, 1080,
         1070, 1070, 1060, 1060, 1050, 1040, 1030, 1010, 1000, 990, 980, 970, 960, 950, 940, 920, 910, 900, 890, 880,
         870, 860, 850, 840, 830, 820, 820, 810, 800, 790, 780, 770, 760, 760, 750, 740, 730, 720, 720, 710, 700, 700,
         690, 680, 680, 670, 660, 660, 650, 640, 640, 630, 630, 620, 620, 610, 610, 600, 600, 590, 590, 580, 580, 570,
         570, 560, 560, 550, 550, 540, 540, 540, 530, 530, 520, 520, 520, 510, 510, 510, 500, 500, 490, 490, 490, 480,
         480, 480, 480, 470, 470, 470, 460, 460, 460, 450, 450, 450, 450, 440, 440, 440, 440, 430, 430, 430, 430, 420,
         420, 420, 420, 410, 410, 410, 410, 410, 400, 400, 400, 400, 390, 390, 390, 390, 390, 380, 380, 380, 380, 380,
         370, 370, 370, 370, 370, 360, 360, 360, 360, 350, 350, 350, 340])

    # Start simulation
    for i in tqdm(range(len(gamma))):
        angle = gamma[i]
        for j, vel in enumerate(v0):
            if vel < min_v[i]:
                continue

            if not mass_diagnostics:
                mr = launch_circular(angle, vel, Isp=Isp, return_mass_ratio=True)
            else:
                mr, mp_end, dm_manoeuvre, height, gamma_fin, velocity = launch_circular(angle, vel, Isp=Isp, return_mass_ratio=True, full_output=True)
                end_propellant[i, j] = mp_end
                circularization_mass[i, j] = dm_manoeuvre
                final_height[i, j] = height
                final_angle[i, j] = gamma_fin
                final_vel[i, j] = velocity

            mass_ratio[i, j] = mr

    if mass_diagnostics:
        return mass_ratio, end_propellant, circularization_mass, final_height, final_angle, final_vel

    if plot_mass_ratio:
        height, width = mass_ratio.shape
        if height > width:
            repeat_factor = height // width
            stretched_mass_ratio = np.repeat(mass_ratio, repeat_factor, axis=1)
            stretched_mass_ratio = stretched_mass_ratio[:, :height]
        else:
            repeat_factor = width // height
            stretched_mass_ratio = np.repeat(mass_ratio, repeat_factor, axis=0)
            stretched_mass_ratio = stretched_mass_ratio[:width, :]

        # filled_mr = fill_nan_2d(stretched_mass_ratio)
        np.savetxt('stretched_mass_ratio.csv', stretched_mass_ratio, delimiter=',')

        plt.imshow(stretched_mass_ratio)
        cbar = plt.colorbar()
        cbar.set_label('Mass ratio')

        plt.xticks(ticks=np.arange(0 + int(repeat_factor / 2), len(stretched_mass_ratio[0]), 2 * repeat_factor), labels=np.array(np.round(v0[::2]), dtype=int))
        plt.yticks(ticks=np.arange(0, len(gamma), 2), labels=np.array(np.round(np.degrees(gamma[::2]), 1), dtype=int))

        plt.xlabel('Initial velocity [m/s]')
        plt.ylabel('Launch angle [deg]')
        plt.show()

        return stretched_mass_ratio

    return mass_ratio


def minimum_velocity_to_orbit(
        gamma: np.ndarray = np.radians(np.arange(0.5, 90, 0.5)),
        construction_mass_ratio: float = 0.1,
        Isp: float = 300
        ):
    """
    Function to find the minimum velocity needed to reach orbit from the Moon surface given a specific construction
    mass ratio and specific impulse
    :param gamma: np.ndarray, launch angles to be evaluated
    :param construction_mass_ratio: float, construction mass ratio
    :param Isp: float, specific impulse
    """
    # TODO: check
    velocity = np.zeros(len(gamma))
    ratio = np.zeros(len(gamma))
    v = 100
    for i in tqdm(range(len(gamma))):
        mass_ratio = launch_circular(gamma[-(i+1)], v, Isp=Isp, return_mass_ratio=True)
        while np.isnan(mass_ratio):
            v += 1
            mass_ratio = launch_circular(gamma[-(i+1)], v, Isp=Isp, return_mass_ratio=True)

        velocity[-(i+1)] = v
        ratio[-(i+1)] = mass_ratio

    return velocity, ratio


if __name__ == '__main__':
    # Design range
    gamma = np.radians(np.arange(0.5, 90, 0.5))
    v0 = np.linspace(200, 2000, len(gamma) + 1)

    mass_ratio, end_prop, dm_circ = optimize_mass_ratio(gamma, v0, plot_mass_ratio=True, mass_diagnostics=True)


    # # Load the mass_ratio data
    # mass_ratio = np.loadtxt('filled_mass_ratio.csv', delimiter=',')
    # mass_ratio[mass_ratio <= 0] = np.nan
    #
    # # Create a heatmap
    # plt.imshow(mass_ratio, aspect='auto', origin='lower', cmap='viridis')
    # cbar = plt.colorbar()
    # cbar.set_label('Mass ratio')
    #
    # # Add contour lines
    # contours = plt.contour(mass_ratio, colors='white', linewidths=0.5)
    # plt.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
    #
    # # Find the minimum value in each row and plot a red line pointing at these minimum values
    # min_indices = np.nanargmin(mass_ratio, axis=1)
    # plt.plot(min_indices, np.arange(len(min_indices)), 'r-', linewidth=2)
    #
    # # Set axis labels and title
    # plt.xlabel('Initial velocity [m/s]')
    # plt.ylabel('Launch angle [deg]')
    # # plt.title('Mass ratio heatmap with contours and minimum values')
    #
    # # Set x and y ticks
    # plt.xticks(ticks=np.arange(0, len(v0), 20), labels=np.array(np.round(v0[::20]), dtype=int))
    # plt.yticks(ticks=np.arange(9, len(gamma), 10), labels=np.array(np.round(np.degrees(gamma[9::10]), 1), dtype=int))
    #
    # # Show the plot
    # plt.show()



