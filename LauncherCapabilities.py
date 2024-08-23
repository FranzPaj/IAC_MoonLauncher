'''
Code for analysing the capabilities of different launchers and scenarios. It covers the differences in P/L that can be carries to a given
destination, as well as the Launch window tradeoffs that correspond to different launch methods (Earth launch, Earth refuelling, 
Moon launch, Moon refuelling).
'''

# Generic imports
import numpy as np
import os
import matplotlib.pyplot as plt

# Tudat imports
from tudatpy import constants

# Custom imports
from OrbitalDynamics.pycode.HelperFunctions import DirectPlanetTransfer, DirectLunarTransfer, FarPlanetTransfer, Launcher, average_radius_dict
from OrbitalDynamics.pycode import PlotGenerator as PlotGen

# Get path of current directory
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()



if __name__ == '__main__':

    ###########################################################################
    # READ USEFUL DATA ########################################################
    ###########################################################################

    # Get the directory for the stored yaml data
    yaml_dir = os.path.join(current_dir, 'OrbitalDynamics', 'input', 'launcher_data')
    # Get the paths for the specific launchers
    starship_path = os.path.join(yaml_dir, 'starship.yaml')

    starship = Launcher(starship_path)

    starship_refuelled_deltav_limit = starship.get_available_refuelled_deltav(4000)
    starship_earth_launch_deltav_limit = starship.get_available_earth_launch_deltav(4000)
    deltav_for_lunar_ascent = 1787
    starship_moon_launch_deltav_limit = starship_refuelled_deltav_limit - deltav_for_lunar_ascent

    print('Available DeltaV:')
    print(f'After refuel: {starship_refuelled_deltav_limit/1000:.2f} km/s | After Earth launch: {starship_earth_launch_deltav_limit/1000:.2f} km/s | '
          f'After Moon launch: {starship_moon_launch_deltav_limit/1000:.2f} km/s' )


    # -----------------------------------------------------------------------------------------------

    ###########################################################################
    # MARS CASE ###############################################################
    ###########################################################################

    print('-------------------------------------')
    print('Mars case study')


    ###########################################################################
    # PLOT FIGURES FOR MARS TRANSFER ##########################################
    ###########################################################################

    data_dir = os.path.join(current_dir, 'OrbitalDynamics', 'output', 'mars_transfer')

    # Load the relevant data
    departure_time_ls = np.loadtxt(os.path.join(data_dir, 'departure_time_range.txt'))
    tof_ls = np.loadtxt(os.path.join(data_dir, 'tof_range.txt'))
    deltav_matrix_earth = np.loadtxt(os.path.join(data_dir, 'earth_launch.txt'))
    deltav_matrix_moon = np.loadtxt(os.path.join(data_dir, 'moon_launch.txt'))

    min_deltav_earth = np.min(deltav_matrix_earth)
    min_deltav_moon = np.min(deltav_matrix_moon)
    print('Minimum DeltaV needed for Mars transfer')
    print('Earth:', min_deltav_earth/1000, 'km/s', '| Moon:', min_deltav_moon/1000, 'km/s')

    starship_possible_refuelled_earth_pl = starship.get_mars_possible_refuelled_pl(min_deltav_earth)
    starship_possible_refuelled_moon_pl = starship.get_mars_possible_refuelled_pl(min_deltav_moon)
    starship_possible_moon_launch_pl = starship.get_mars_possible_moon_launch_pl(min_deltav_moon + deltav_for_lunar_ascent)
    print('Possible P/L transportable via starship')
    print(f'Earth refuel: {starship_possible_refuelled_earth_pl / 1000:.2f} t | Moon refuel: {starship_possible_refuelled_moon_pl / 1000:.2f} t |' 
          f' Moon launch: {starship_possible_moon_launch_pl / 1000:.2f} t')

    fig1 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_earth)

    fig2 = PlotGen.porkchop_plot(departure_time_ls, tof_ls, deltav_matrix_moon)

    res_dict = {
        'earth_launch':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_earth,
            'deltav_cutoff': starship_earth_launch_deltav_limit,
            'color': 'green',
            'linestyle': '-',
            'label': 'Earth launch',
        },
        'earth_refuel':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_earth,
            'deltav_cutoff': starship_refuelled_deltav_limit,
            'color': 'green',
            'linestyle': '--',
            'label': 'Earth refuelling',
        },
        'moon_launch':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_moon,
            'deltav_cutoff': starship_moon_launch_deltav_limit,
            'color': 'grey',
            'linestyle': '-',
            'label': 'Moon launch',
        },
        'moon_refuel':{
            'departure_time': departure_time_ls,
            'tof': tof_ls,
            'deltav': deltav_matrix_moon,
            'deltav_cutoff': starship_refuelled_deltav_limit,
            'color': 'grey',
            'linestyle': '--',
            'label': 'Moon refuelling',
        },
    }

    fig3 = PlotGen.feasibility_comparison(res_dict)

    # plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)



    ###########################################################################
    # CHECK TRANSFER CAPABILITIES TO MOON #####################################
    ###########################################################################

    transfer = DirectLunarTransfer()
    sma_dep_earth = average_radius_dict['Earth'] + 200e3  # A 200 km altitude departure orbit
    ecc_dep_earth = 0
    sma_arr_moon = average_radius_dict['Moon'] + 100e3  # A 100 km altitude Moon arrival orbit
    ecc_arr_moon = 0

    deltav_moon_transfer = transfer.get_transfer_deltav(sma_dep_earth, ecc_dep_earth, sma_arr_moon, ecc_arr_moon)

    
    big_lambda = np.exp(deltav_moon_transfer / (starship.Isp_f * starship.g0))
    pl_to_moon = starship.prop_mass_f / (big_lambda - 1) - starship.dry_mass_f

    print(f'P/L that a refuelled Starship can transport to LLO: {pl_to_moon/1000:.2f} t')


    ###########################################################################
    # HANDLE REFUELING WITH LIMITED OXIDANT ###################################
    ###########################################################################

    num_points = 550
    pl_ls = np.linspace(1e3, 550e3, num_points)  # Cover the 1t - 550t interval with a 1t density
    oxy_ls = np.zeros(num_points)
    oxy_ls[:] = np.nan
    fuel_supply_needed_ls = np.zeros(num_points)
    frac_ls = np.zeros(num_points)

    o2_to_ch4_mass_ratio = 4

    # Data needed
    liq = starship.prop_mass_f
    dry = starship.dry_mass_f
    c = starship.Isp_f * starship.g0
    exp_fact_1 = np.exp(deltav_moon_transfer / c)
    exp_fact_2 = np.exp(min_deltav_moon / c)

    best_pl = 0

    for index, pl in enumerate(pl_ls):

        coeff_mat = np.array([[liq * exp_fact_1, exp_fact_1 - 1], [liq, (1 + o2_to_ch4_mass_ratio)]])
        val = np.array([liq - (dry + pl) * (exp_fact_1 - 1), (dry + pl) * (exp_fact_2 - 1)])
        sol = np.linalg.solve(coeff_mat, val)
        remaining_prop_mass_fract = sol[0]
        oxy_mass = sol[1]

        if oxy_mass < 0:
            oxy_mass = 0
            
        if remaining_prop_mass_fract < 0:
            oxy_mass = np.nan
        else:
            best_pl = pl

        oxy_ls[index] = oxy_mass
        frac_ls[index] = remaining_prop_mass_fract
        fuel_supply_needed_ls[index] = 4 * oxy_mass

    print(f'Biggest Payload to Mars if oxydant needs to be brought as well: {best_pl / 1000:.2f} t')

    # plt.plot(pl_ls/1000, oxy_ls/1000)
    # plt.show()
    # plt.plot(pl_ls/1000, fuel_supply_needed_ls/1000)
    # plt.show()


    ###########################################################################
    # MASS ESTIMATION FOR PAYLOAD DIRECTLY LAUNCHED WITH ALT METHODS ##########
    ###########################################################################

    nom_pl = 4e3  # kg
    epsilon = 0.05  # Construction ratio
    deltav = min_deltav_moon + 800
    exp_fact = np.exp(deltav / c)
    prop_mass = nom_pl * (exp_fact - 1) / (1 / (1 - epsilon) - epsilon / (1 - epsilon) * exp_fact)

    mass_to_be_yeeted = prop_mass * epsilon / (1 - epsilon) + prop_mass + nom_pl
    print(f'Approximate mass to be yeeted to LLO for a 4 t mission to Mars: {mass_to_be_yeeted/1000:.2f} t')


    ###########################################################################
    # URANUS CASE STUDY #######################################################
    ###########################################################################

    print('----------------------------------------------------------------------------')
    print('Uranus case study')

    # Get Moon DeltaV for given Vinf
    c3 = 12.04 * 10**6  # m^2/s^2
    transfer = FarPlanetTransfer(c3)
    sma_earth_dep = average_radius_dict['Earth'] + 200e3  # 200 km altitude orbit - m
    ecc_earth_dep = 0
    deltav_uranus_from_earth = transfer.get_earth_departure_deltav(sma_earth_dep, ecc_earth_dep)
    sma_moon_dep = average_radius_dict['Moon'] + 100e3  # 100 km altitude orbit - m
    ecc_moon_dep = 0
    deltav_uranus_from_moon = transfer.get_moon_departure_deltav(sma_moon_dep, ecc_moon_dep)

    print('DeltaV needed for Uranus transfer:')
    print(f'Earth: {deltav_uranus_from_earth/1000:.2f} km/s | Moon: {deltav_uranus_from_moon/1000:.2f} km/s')


