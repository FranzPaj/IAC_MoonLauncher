from OrbitalDynamics.launch_trajectories import moon_launch_optim
from OrbitalDynamics.spinLaunch import optimize_mass_ratio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def polynomial(x, *p):
    return sum([p[i] * x ** i for i in range(len(p))])


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
    gamma = np.radians(np.arange(0.1, 10, 0.1))
    v0 = np.arange(1200, 2000, 50)
    spinlaunch_ratio_0, optimal_v = optimize_mass_ratio(gamma, v0, Isp, plot_mass_ratio=False)

    print(spinlaunch_ratio_0)
    print(gamma)

    gamma_1 = np.radians(np.arange(10, 90, 5))
    v1 = np.arange(800, 2000, 100)
    spinlaunch_ratio_1, optimal_v_1 = optimize_mass_ratio(gamma_1, v1, Isp, plot_mass_ratio=False)

    # TODO: best fit curve for the first 10-15 degrees of spinlaunch ratio
    popt, _ = curve_fit(polynomial, gamma, spinlaunch_ratio_0, p0=[1] * 4)
    n_polynomial = lambda: polynomial(gamma, *popt)

    # Compare the results with all systems.
    # TODO: plot best fit curve for gamma < 10 and normal curve for gamma > 10
    plt.plot(np.degrees(launch_params_maglev[:, 0]), maglev_ratio, label='Maglev')
    plt.plot(np.degrees(gamma), n_polynomial(), label='SpinLaunch', color='tab:orange')
    plt.plot(np.degrees(gamma_1), spinlaunch_ratio_1, color='tab:orange')
    plt.scatter(np.degrees(gamma), spinlaunch_ratio_0, color='tab:orange', marker='o')
    plt.axhline(y=baseline_ratio, color='r', linestyle='--', label='baseline')
    plt.xlabel('Launch angle [deg]')
    plt.ylabel('Mass ratio')
    plt.legend()
    plt.show()
