
# Generic imports
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt

# Pygmo imports
import pygmo as pg

# Custom imports
from OrbitalDynamics.pycode.HelperFunctions import Launcher, average_radius_dict, gravitational_param_dict
from OrbitalDynamics import spinLaunch

# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()

def polar_int_fun(t, state, thrust, g_moon, g0, Isp, t_vert):
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
    # Condition for initiation of gravity turn
    if t < t_vert:
        gammadot = 0
    else:
        gammadot = - 1 / v * g_moon * np.cos(gamma) * (1 - v**2 / (g_moon * r))  # From Re-entry Systems - I'll share the sauce later, Javi
    vdot = thrust / mtot - g_moon * np.sin(gamma)  # From Rocket Motion (and simple dynamics)
    mdot = - thrust / (Isp * g0)  # Better than Tsiolkovskji, does not require integration with time

    state_dot = np.array([rdot, thetadot, gammadot, vdot, mdot, mdot])

    return state_dot


class GravityTurnOptimalKickoffProblem:

    def __init__(self, launcher: Launcher, h_th, mass_pl,  decision_variable_range):
        """
        Constructor for the GravityTurnProblem class.
        """

        # Copy arguments as attributes
        self.stage_mass = launcher.dry_mass_f + launcher.prop_mass_f
        self.total_starting_mass = self.stage_mass + mass_pl
        self.total_starting_prop_mass = launcher.prop_mass_f
        self.thrust = launcher.thrust_f
        self.g0 = 9.81
        self.Isp = launcher.Isp_f
        self.g_moon = 1.625  # Gravity on the Moon

        self.h_th = h_th
        self.decision_variable_range = decision_variable_range

    def get_bounds(self):
        return self.decision_variable_range

    # Return number of objectives
    def get_nobj(self):
        return 1
    
    # Return function name
    def get_name(self):
        return "Optimal gravity turn kickoff optimisation"
    
    def get_orbit_reaching_condition(self, gamma0, burnt_prop_fraction, t_vert):
        '''
        Function for finding the state with which the spacecraft reaches a given altitude. Useful for getting
        needed quantities for the final propellant ratio.
        '''

        mtot0 = self.total_starting_mass  # Total mass at t0
        mp0 = self.total_starting_prop_mass * burnt_prop_fraction  # Propellant to be burnt during ascent
        r0 = average_radius_dict['Moon'] + 1e-6  # Radius at t0 - m
        theta0 = 0  # Polar angle at t0
        v0 = 0  # Static start

        # Integrate EoMs
        # Define stop condition when rocket reaches apogee
        def stop_condition(t, state, thrust, g_moon, g0, Isp, t_vert):
            return state[0] - (average_radius_dict['Moon'] + self.h_th)
        stop_condition.terminal = True

        # Define arguments - constants useful during the integration
        args = (self.thrust, self.g_moon, self.g0, self.Isp, t_vert)
        # Define parameters for integration
        time_span_orbit = [0, 24 * 3600]
        tol = 1.0e-12 # Relative and absolute tolerance for integration
        # Define initial state
        x0 = np.array([r0, theta0, gamma0, v0, mtot0, mp0])
        sol = scipy.integrate.solve_ivp(polar_int_fun, time_span_orbit, x0, method='RK45',
                                        rtol=tol, atol=tol, args=args, events=stop_condition)

        y = sol.y.T  # Get history of state vectors
        h_fin = y[-1,0] - average_radius_dict['Moon']  # Get altitude over Moon surface
        mtot_fin = y[-1, 4]  # Get final mass
        v_fin = y[-1, 3]  # Get final velocity
        gamma_fin = y[-1,2]  # Get final flight-path anlge
        mp_fin = y[-1,5]  # Get final propellant-to-be-burnt mass

        return h_fin, v_fin, gamma_fin, mtot_fin, mp_fin

    def fitness(self, trajectory_parameters) -> float:
        
        # Unpack decision variables
        gamma0 = trajectory_parameters[0]
        burnt_prop_fraction = trajectory_parameters[1]
        t_vert = trajectory_parameters[2]

        # Get condition when reaching threshold altitude
        h_fin, v_fin, gamma_fin, mtot_fin, mp_fin = self.get_orbit_reaching_condition(gamma0, burnt_prop_fraction, t_vert)

        # Calculate deltaV for executing the gravity turn
        deltav_turn = self.Isp * self.g0 * np.log(self.total_starting_mass / mtot_fin)

        # Calculate manoeuvre to enter circular Lunar orbit
        v_target = np.sqrt(gravitational_param_dict['Moon'] / (average_radius_dict['Moon'] + self.h_th))  # Orbital velocity at 100 km
        deltav_manoeuvre = np.sqrt(v_fin ** 2 + v_target ** 2 - 2 * v_fin * v_target * np.cos(gamma_fin))  # From cosine law

        # Check if reached the orbit
        if h_fin < self.h_th * 95 / 100:
            fitness = 1e9
        else:
            fitness = deltav_turn + deltav_manoeuvre

        return [fitness]
    

def main():

    # Get the directory for the stored yaml data
    yaml_dir = os.path.join(current_dir, 'input', 'launcher_data')
    # Get the paths for the specific launchers
    starship_path = os.path.join(yaml_dir, 'starship.yaml')

    starship = Launcher(starship_path)

    # Get time for motor burnout
    mdot = starship.thrust_f / (starship.Isp_f * starship.g0)
    tb = starship.prop_mass_f / mdot  # Burnout time of engines if all propellant was spent

    # Define decision variable range
    minimum_gamma = 0
    maximum_gamma = np.deg2rad(90)
    minimum_burnt_prop_ratio = 0
    maximum_burnt_prop_ratio = 1
    minimum_time_vertical = 0
    maximum_time_vertical = tb
    decision_variable_range = \
        [[minimum_gamma, minimum_burnt_prop_ratio, minimum_time_vertical],
         [maximum_gamma, maximum_burnt_prop_ratio, maximum_time_vertical]]
    
    ##############################################################################
    # CONDUCT OPTIMISATION #######################################################
    ##############################################################################

    # Define target altitude
    h_th = 100e3  # m
    # Define target payload
    mass_pl = 4e3  # kg
    mass_pl = 210.46e3  # kg
    # Define current seed and population size
    current_seed = 42
    pop_size = 150

    #### Define the problem class ####
    prob = pg.problem(GravityTurnOptimalKickoffProblem(starship, h_th, mass_pl, decision_variable_range))

    #### Define the algorithm ####
    algo = pg.algorithm(pg.de(seed = current_seed))
    algo.set_verbosity(10)  # Print every n evolutions

    #### Create the population #### 
    pop = pg.population(prob, size=pop_size, seed=current_seed)

    fitness_evolution_ls = list()
    old_average_fitness = 1
    generation_number = 0
    generations_without_change = 0
    # Define criteria for convergence condition and maximum number of generations
    convergence_threshold = 0.001
    maximum_number_of_equilibrium_generations = 4
    maximum_generation_number = 80

    while generations_without_change < maximum_number_of_equilibrium_generations and generation_number < maximum_generation_number:
        print('-----------------------------------------------------------')
        print(f'Generation {generation_number} evolving')
        pop = algo.evolve(pop)
        fitnesses = pop.get_f()
        new_average_fitness = np.mean(fitnesses)
        fitness_evolution_ls.append(new_average_fitness)
        variation = np.abs((new_average_fitness - old_average_fitness) / old_average_fitness)
        if variation < convergence_threshold:
            generations_without_change = generations_without_change + 1
            print(f'{generations_without_change} generations without significant change')
        else:
            generations_without_change = 0
            print('Improved result:', new_average_fitness)
        print('Variation:', variation * 100, '%')
        old_average_fitness = new_average_fitness
        generation_number = generation_number + 1

    parameters = np.array(pop.get_x())
    fitnesses = np.array(pop.get_f())
    # print('Gamma0:', np.rad2deg(parameters[0]), '| Burnt propellant ratio:', parameters[1],'| Time:', parameters[2], 's')

    # Extract result
    plt.plot(range(generation_number), fitness_evolution_ls,) 
    plt.show() 
    plt.scatter(np.rad2deg(parameters[:,0]), fitnesses)
    plt.show()
    plt.scatter(parameters[:,1] * 100, fitnesses)
    plt.show()
    plt.scatter(parameters[:,2] / 60 , fitnesses)
    plt.show()
