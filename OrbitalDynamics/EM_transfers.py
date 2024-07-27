# Standard imports
from pycode.HelperFunctions import Orbit, gravitational_param_dict, average_radius_dict, sma_moon
import matplotlib.pyplot as plt
import numpy as np

# Tudat modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime


if __name__ == '__main__':
    # Earth Moon 2D transfer
    # TODO: Implement polar transfer

    # Define initial parameters
    h_origin = 100e3  # m
    body_origin = 'Moon'

    v_moon = np.sqrt(gravitational_param_dict['Earth'] / sma_moon)
    SOI_moon = 66100e3  # m

    mu0 = gravitational_param_dict[body_origin]
    R0 = average_radius_dict[body_origin]

    # Define target parameters
    h_target = 300e3  # m
    body_target = 'Earth'
    muE = gravitational_param_dict[body_target]
    Re = average_radius_dict[body_target]

    # Define transfer orbit
    sma_transfer = (sma_moon + (R0 + h_origin) + (Re + h_target)) / 2
    e_transfer = 1 - (Re + h_target) / sma_transfer

    transfer_orbit = Orbit('Earth', sma_transfer, e_transfer, inclination=0)

    # Calculate parameters at the border of the SOI
    theta_SOI = np.arccos((transfer_orbit.p / transfer_orbit.sma - 1) / e_transfer)  # Where the fuck did this ccome from??
    r_SOI = np.sqrt(sma_moon**2 + SOI_moon**2 - 2 * sma_moon * SOI_moon * np.cos(np.pi - theta_SOI))
    v_SOI = np.sqrt(2 * (muE / r_SOI - muE / (2 * sma_transfer)))
    gamma_SOI = np.arccos(transfer_orbit.h / (r_SOI * v_SOI))

    # Moon escape manoeuvre
    v_inf = v_SOI * np.array([[np.cos(gamma_SOI)], [np.sin(gamma_SOI)]]) - np.array([[0], [v_moon]])
    v_inf = np.linalg.norm(v_inf)

    v0 = np.sqrt(mu0 / (R0 + h_origin))
    deltaV = np.sqrt(2 * (v_inf**2 / 2 + mu0 / (R0 + h_origin))) - v0

    # capture
    total_deltaV = deltaV + np.abs(transfer_orbit.vp - np.sqrt(muE / (Re + h_target)))
    print(f'Delta-V for Moon escape: {deltaV:.2f} m/s')
    print(f'Delta-V for capture: {total_deltaV - deltaV:.2f} m/s')
    print(f'Total Delta-V: {total_deltaV / 1e3:.3f} km/s')

    ### tudat validation ###

    # Load spice kernels
    spice.load_standard_kernels()
    simulation_start_epoch = DateTime(2000, 4, 25).epoch()
    simulation_end_epoch = simulation_start_epoch + 30 * constants.JULIAN_DAY

    # Create environment
    bodies_to_create = ["Sun", "Earth", "Moon"]
    body_settings = environment_setup.get_default_body_settings(bodies_to_create)
    system_of_bodies = environment_setup.create_system_of_bodies(body_settings)

    system_of_bodies.create_empty_body("Vehicle")
    system_of_bodies.get_body("Vehicle").set_constant_mass(1000)

    # Define thrust and guidance settings
    rotation_model_settings = environment_setup.rotation_model.orbital_state_direction_based(
        central_body="Moon",
        is_colinear_with_velocity=True,
        direction_is_opposite_to_vector=False,
        base_frame="",
        target_frame="VehicleFixed")
    environment_setup.add_rotation_model(system_of_bodies, 'Vehicle', rotation_model_settings)

    thrust_magnitude_settings = (
        propagation_setup.thrust.constant_thrust_magnitude(
            thrust_magnitude=0.0, specific_impulse=5e3))
    environment_setup.add_engine_model(
        'Vehicle', 'MainEngine', thrust_magnitude_settings, system_of_bodies)

    # Propagation setup
    bodies_to_propagate = ["Vehicle"]
    central_bodies = ["Moon"]

    # Acceleration model
    acceleration_on_vehicle = dict(
        Vehicle=[propagation_setup.acceleration.thrust_from_engine('MainEngine')],
        Earth = [propagation_setup.acceleration.point_mass_gravity()],
        Moon = [propagation_setup.acceleration.point_mass_gravity()],
        Sun = [propagation_setup.acceleration.point_mass_gravity()]
    )

    acceleration_dict = dict(Vehicle=acceleration_on_vehicle)
    acceleration_models = propagation_setup.create_acceleration_models(
        body_system=system_of_bodies,
        selected_acceleration_per_body=acceleration_dict,
        bodies_to_propagate=bodies_to_propagate,
        central_bodies=central_bodies
    )

    # Define initial state
    moon_state = spice.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "None", simulation_start_epoch)

    # TODO: check initial conditions
    satellite_original_position = moon_state[:3] * (1 + (R0 + h_origin) / np.linalg.norm(moon_state[:3]))
    # v_sat = -2560 * moon_state[3:] / np.linalg.norm(moon_state[3:]) + moon_state[3:]
    v_sat = -transfer_orbit.va * moon_state[3:] / np.linalg.norm(moon_state[3:])

    # system_initial_state = np.array([satellite_original_position[0],
    #                                  satellite_original_position[1],
    #                                  satellite_original_position[2],
    #                                  v_sat[0],
    #                                  v_sat[1],
    #                                  v_sat[2]])

    system_initial_state = np.array([R0 + h_origin, 0.0, 0.0, 0.0, -v0 - 720, 0.0])

    # Create dependent variables
    # vehicle_altitude_dep_var = propagation_setup.dependent_variable.altitude("Vehicle", "Earth")
    vehicle_altitude_dep_var = propagation_setup.dependent_variable.altitude("Vehicle", "Earth")
    dependent_variables_to_save = [vehicle_altitude_dep_var]

    # Create termination settings
    termination_distance_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=vehicle_altitude_dep_var,
        limit_value=average_radius_dict['Earth']+h_target,
        use_as_lower_limit=True)

    termination_time_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    termination_settings = propagation_setup.propagator.hybrid_termination(
        [termination_distance_settings, termination_time_settings],
        fulfill_single_condition=True)

    # Integrator settings
    initial_time_step = 10.0
    minimum_time_step = 0.01
    maximum_time_step = 86400

    tolerance = 1e-10
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        initial_time_step,
        propagation_setup.integrator.rkf_78,
        minimum_time_step,
        maximum_time_step,
        relative_error_tolerance=tolerance,
        absolute_error_tolerance=tolerance)

    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        system_initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        propagation_setup.propagator.cowell,
        output_variables=[vehicle_altitude_dep_var]
    )

    propagator_settings = propagation_setup.propagator.multitype(
        [translational_propagator_settings],
        integrator_settings,
        simulation_start_epoch,
        termination_settings,
        [vehicle_altitude_dep_var])


    # Propagate orbit
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        system_of_bodies, propagator_settings)

    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    vehicle_array = result2array(state_history)
    dep_var_array = result2array(dependent_variable_history)


    # Get the Moon state
    # moon_states_from_spice = {
    #     epoch: spice.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "None", epoch)
    #     for epoch in list(state_history.keys())
    # }

    earth_states_from_spice = {
        epoch: spice.get_body_cartesian_state_at_epoch("Earth", "Moon", "J2000", "None", epoch)
        for epoch in list(state_history.keys())
    }

    earth_array = result2array(earth_states_from_spice)

    # moon_array = result2array(moon_states_from_spice)

    # Plot stuff
    time_days = (vehicle_array[:, 0] - vehicle_array[0, 0]) / constants.JULIAN_DAY

    # Height vs days
    fig1 = plt.figure(figsize=(9, 5))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(f"Vehicle altitude above Earth")

    ax1.plot(time_days, dep_var_array[:, 1] / 1e3)
    ax1.set_xlabel("Simulation time [day]")
    ax1.set_ylabel("Vehicle altitude [km]")
    ax1.grid()

    fig1.tight_layout()
    fig1.show()

    # Trajectory
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_title(f"System state evolution in 3D")

    ax3.plot(vehicle_array[:, 1], vehicle_array[:, 2], vehicle_array[:, 3], label="Vehicle", linestyle="-",
             color="green")
    # ax3.plot(moon_array[:, 1], moon_array[:, 2], moon_array[:, 3], label="Moon", linestyle="-", color="grey")
    ax3.plot(earth_array[:, 1], earth_array[:, 2], earth_array[:, 3], label="Earth", linestyle="-", color="grey")
    ax3.scatter(0.0, 0.0, 0.0, label="Moon", marker="o", color="blue")

    ax3.legend()
    # ax3.set_xlim([-3E8, 3E8]), ax3.set_ylim([-3E8, 3E8]), ax3.set_zlim([-3E8, 3E8])
    ax3.set_xlim([-4E8, 4E8]), ax3.set_ylim([-4E8, 4E8]), ax3.set_zlim([-4E8, 4E8])
    ax3.set_xlabel("x [m]"), ax3.set_ylabel("y [m]"), ax3.set_zlabel("z [m]")

    fig3.tight_layout()
    fig3.show()
