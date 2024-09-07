import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.analyze import sobol


def arm_mass(takeoff_mass, density, yield_strength, launch_speed, safety_factor):
    return takeoff_mass * (np.exp(density / yield_strength * launch_speed ** 2 * safety_factor) - 1)


def arm_area(z, takeoff_mass, density, yield_strength, launch_speed, arm_length, safety_factor):
    return (takeoff_mass / yield_strength * launch_speed ** 2 / arm_length * safety_factor *
            np.exp(density / yield_strength * launch_speed ** 2 / arm_length * safety_factor * z))


def arm_angular_moment_of_inertia(takeoff_mass, density, yield_strength, launch_speed, arm_length, safety_factor, n_elem=50):
    radii = np.linspace(0, arm_length, n_elem)
    dr = radii[1] - radii[0]
    masses = density * arm_area(
        radii, takeoff_mass, density, yield_strength,
        launch_speed, arm_length, safety_factor
    ) * dr
    return np.sum(masses * radii**2)


if __name__ == '__main__':
    g_0 = 9.80665   # [m/s2]
    max_load = 10000 * g_0      # [g]

    M_TO = np.linspace(10, 10000, 100)     # [kg]
    rho = 1700          # [kg/m3] CFRP
    sigma_y = 800e6     # [Pa] CFRP
    v_launch = np.linspace(100, 2000, 100)     # [m/s]
    SF = 1.65      # [-]

    xv1, yv1 = np.meshgrid(v_launch, M_TO)
    M_arm = arm_mass(yv1, rho, sigma_y, xv1, SF)

    plt.contourf(v_launch, M_TO, M_arm, locator=ticker.LogLocator())

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$v_{launch}$ [m/s]")
    plt.ylabel(r"$M_{TO}$ [kg]")

    plt.colorbar(label="$M_{arm}$ [kg]")

    plt.show()

    # TODO: power
    M_TO_max = 100
    L_arm = np.linspace(1, 50, 100)
    xv2, yv2 = np.meshgrid(v_launch, L_arm)
    arm_ammoi = arm_angular_moment_of_inertia(M_TO, rho, sigma_y, xv2, yv2, SF)
    payload_ammoi = M_TO_max * yv2 ** 2
    total_ammoi = 2 * (arm_ammoi + payload_ammoi)

    time_to_launch = 60 * 60    # [s]
    final_angular_velocity = xv2 / yv2
    power_required = total_ammoi * final_angular_velocity**2 / (2 * time_to_launch)

    plt.contourf(v_launch, L_arm, power_required, locator=ticker.LogLocator())

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$v_{launch}$ [m/s]")
    plt.ylabel(r"$L_{arm}$ [m]")

    plt.colorbar(label="$P_{req}$ [W]")

    plt.show()

    # TODO: g-load
    g_load = xv2 ** 2 / yv2 / 9.81

    plt.contourf(v_launch, L_arm, g_load, locator=ticker.LogLocator())

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$v_{launch}$ [m/s]")
    plt.ylabel(r"$L_{arm}$ [m]")

    plt.colorbar(label="g load [g]")

    plt.show()

    # TODO: sensitivity
    problem = {'num_vars': 5,
               'names': ['takeoff_mass', 'density', 'yield_strength', 'launch_speed', 'safety_factor'],
               'bounds': [[10, 1000],
                          [1500, 1900],
                          [600e6, 1000e6],
                          [500, 2000],
                          [2.3, 3]]}

    param_values = saltelli.sample(problem, 2 ** 6)

    arm_masses = np.array([arm_mass(*params) for params in tqdm(param_values)])

    print(f'Total sample size = {len(arm_masses)}')

    param_values = param_values[~np.isnan(arm_masses)]
    arm_masses = arm_masses[~np.isnan(arm_masses)]

    print(f'Filtered sample size = {len(arm_masses)}')

    while len(arm_masses) % (2 * problem['num_vars'] + 2) != 0:
        arm_masses = arm_masses[:-1]
        param_values = param_values[:-1]

    print('Reduced sample size =', len(arm_masses))

    # Analyze results
    sobol_indices = sobol.analyze(problem, arm_masses, print_to_console=True)

    S1s = sobol_indices['S1']  # First order indices
    STs = sobol_indices['S2']  # Total indices

    print('First-order sobol indices:')
    print(S1s, '\n')

    print('Total sobol indices:')
    print(STs)
