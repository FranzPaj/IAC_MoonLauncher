from OrbitalDynamics.launch_trajectories import moon_launch_optim
from OrbitalDynamics.spinLaunch import optimize_mass_ratio
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # TODO: look more closely at spinlaunch for gamma < 10 deg
    # Rocket parameters
    Isp = 300  # s

    # Baseline
    baseline_ratio = np.exp(1.8e3 / (Isp * 9.81)) - 1

    # Maglev
    deltaV_maglev, launch_params_maglev = moon_launch_optim(return_params=True)
    maglev_ratio = np.exp(deltaV_maglev / (Isp * 9.81)) - 1

    # SpinLaunch
    gamma = np.radians(np.arange(0, 90, 1))
    v0 = np.arange(200, 2000, 100)
    spinlaunch_ratio, optimal_v = optimize_mass_ratio(gamma, v0, Isp, plot_mass_ratio=True)

    # Compare the results with all systems.
    plt.plot(np.degrees(launch_params_maglev[:, 0]), maglev_ratio, label='Maglev')
    plt.plot(np.degrees(gamma), spinlaunch_ratio, label='SpinLaunch')
    plt.axhline(y=baseline_ratio, color='r', linestyle='--', label='baseline')
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Mass ratio')
    plt.legend()
    plt.show()
