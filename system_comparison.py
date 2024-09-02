from OrbitalDynamics.launch_trajectories import moon_launch_optim
from OrbitalDynamics.spinLaunch import fill_nan_2d, optimize_mass_ratio
import matplotlib.pyplot as plt
import numpy as np


def polynomial(x, *p):
    return sum([p[i] * x ** i for i in range(len(p))])


def plot_2D_figure(
        array: np.ndarray,
        colorbar_label: str,
        initial_angles: np.ndarray = np.radians(np.arange(0.5, 90, 0.5)),
        initial_velocities: np.ndarray = np.linspace(200, 2000, 180),
        xlabel: str = 'Launch angle [deg]',
        ylabel: str = 'Initial velocity [m/s]',
        extended: bool = False,
        plot_paths: bool = False,
        contour: bool = False,
        save: str = None):

    if extended:
        array_ext = np.empty((array.shape[0] * 2, array.shape[1]))
        array_ext[::2] = array
        array_ext[1::2] = array

        array = array_ext.T

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a heatmap
    cax = ax.imshow(array, aspect='auto', origin='lower', cmap='viridis')

    # Create color bar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    if contour:
        if extended:
            contours = ax.contour(array, colors='white', linewidths=1)
            ax.clabel(contours, inline=True, fontsize=12, fmt='%1.2f')
        else:
            contours = ax.contour(array, colors='white', linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')

    # Create labels
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # Formatting
    x_labels = np.array(np.arange(10, 90, 10), dtype=int)
    x_ticks = np.searchsorted(initial_angles, np.radians(x_labels)) * 2
    plt.xticks(ticks=x_ticks,
               labels=x_labels,
               fontsize=14)

    y_labels = np.arange(200, 2000, 200)
    y_ticks = np.searchsorted(initial_velocities, y_labels)
    plt.yticks(ticks=y_ticks,
               labels=y_labels,
               fontsize=14)

    # Plot optimal path
    if plot_paths:
        if extended:
            ax.plot(np.arange(len(min_prop_ext)), min_prop_ext,
                    color='tab:red', linestyle='-', label='Case 2 opt')
            ax.plot(np.searchsorted(gamma, launch_params_maglev[:, 0]) * 2,
                    np.searchsorted(v0, launch_params_maglev[:, 1]),
                    color='tab:red', linestyle='--', label='Case 1 opt')
        else:
            ax.plot(np.arange(len(min_prop)), min_prop,
                    color='tab:red', linestyle='-', label='Csae 2 opt')
            ax.plot(np.searchsorted(gamma, launch_params_maglev[:, 0]),
                    np.searchsorted(v0, launch_params_maglev[:, 1]),
                    color='tab:red', linestyle='--', label='Case 1 opt')
        fig.legend(bbox_to_anchor=(0.3, 0.31), fontsize=14)

    ax.grid(True)
    fig.tight_layout()

    if save is not None:
        plt.savefig(f'Plots\\{save}.pdf')
    plt.show()


if __name__ == '__main__':
    # Rocket parameters
    Isp = 300  # s

    # Baseline
    baseline_ratio = np.exp(1.8e3 / (Isp * 9.81)) - 1

    ### Maglev
    deltaV_maglev, launch_params_maglev = moon_launch_optim(return_params=True)
    maglev_ratio = np.exp(deltaV_maglev / (Isp * 9.81)) - 1
    maglev_useful_mass = (900 - 100 * maglev_ratio) / (maglev_ratio + 1)
    maglev_gear_ratio = 900 / maglev_useful_mass

    ### SpinLaunch
    gamma = np.radians(np.arange(0.5, 90, 0.5))
    v0 = np.linspace(200, 2000, len(gamma) + 1)

    # Calculate ratio
    # spinLaunch_ratio_0, end_prop, circularization_prop, final_height, final_angle, final_velocity = optimize_mass_ratio(gamma, v0, Isp, plot_mass_ratio=False, mass_diagnostics=True)
    # np.savetxt('filled_mass_ratio_v2.csv', spinLaunch_ratio_0, delimiter=',')
    # np.savetxt('end_prop_v2.csv', end_prop, delimiter=',')
    # np.savetxt('circularization_prop_v2.csv', circularization_prop, delimiter=',')
    # np.savetxt('final_height.csv', final_height, delimiter=',')
    # np.savetxt('final_angle.csv', final_angle, delimiter=',')
    # mass_ratio = spinLaunch_ratio_0

    # Load values and fill in NaNs
    mass_ratio = np.loadtxt('filled_mass_ratio_v2.csv', delimiter=',')
    mass_ratio[mass_ratio <= 0] = np.nan
    minimum_ratio = np.nanmin(mass_ratio)
    mass_ratio = fill_nan_2d(mass_ratio)
    mass_ratio[mass_ratio <= minimum_ratio] = np.nan

    useful_mass = (900 - 100 * mass_ratio) / (mass_ratio + 1)
    gear_ratio = 900 / useful_mass

    min_prop = np.nanargmin(mass_ratio, axis=1)
    min_prop_ext = np.empty((min_prop.size * 2))
    min_prop_ext[::2] = min_prop
    min_prop_ext[1::2] = min_prop

    end_prop = np.loadtxt('end_prop_v2.csv', delimiter=',')
    end_prop[end_prop <= 0] = np.nan
    minimum_prop = np.nanmin(end_prop)
    end_prop = fill_nan_2d(end_prop)
    end_prop[end_prop <= minimum_prop] = np.nan

    circularization_prop = np.loadtxt('circularization_prop_v2.csv', delimiter=',')
    circularization_prop[circularization_prop <= 0] = np.nan
    minimum_prop = np.nanmin(circularization_prop)
    circularization_prop = fill_nan_2d(circularization_prop)
    circularization_prop[circularization_prop <= minimum_prop] = np.nan


    # Plot spinLaunch
    plot_2D_figure(gear_ratio, 'Gear ratio [-]', extended=True, plot_paths=True,
                   save='gear_ratio', contour=True)

    ### Optimum and minimum velocities and their corresponding ratios
    gear_ratio_T = gear_ratio.T
    first_non_nan_indices = np.full(gear_ratio_T.shape[1], np.nan)

    # Iterate over each column to find the first non-NaN value
    ratio_min_v = []
    min_v = []
    for col in range(gear_ratio_T.shape[1]):
        non_nan_indices = np.where(~np.isnan(gear_ratio_T[:, col]))[0]
        if non_nan_indices.size > 0:
            first_non_nan_indices[col] = int(non_nan_indices[0])
            ratio_min_v.append(gear_ratio_T[int(non_nan_indices[0]), col])
            min_v.append(v0[int(non_nan_indices[0])])

    ## Plot ratios
    fig, ax = plt.subplots(figsize=(12, 6))

    # Maglev params
    ax.plot(np.degrees(launch_params_maglev[:, 0]), maglev_gear_ratio,
            label=r'Case 1 opt', color='tab:blue', linestyle='-')

    # SpinLaunch params
    ax.plot(np.degrees(gamma), np.nanmin(gear_ratio, axis=1),
            label='Case 2 opt', color='tab:blue', linestyle='--')

    # Minimum params
    ax.plot(np.degrees(gamma), ratio_min_v, label='Minimum', color='tab:blue', linestyle=':')

    ax.set_xlabel('Launch angle [deg]', fontsize=14)
    ax.set_ylabel('Gear ratio [-]', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True)
    ax.legend(bbox_to_anchor=(0.64, 0.945), fontsize=12)

    plt.savefig('Plots\\Opt_and_min_gear_ratios.pdf')
    plt.show()

    ## Plot velocities
    fig, ax = plt.subplots(figsize=(12, 6))

    # Maglev params
    ax.plot(np.degrees(launch_params_maglev[:, 0]), launch_params_maglev[:, 1],
            label=r'Case 1 opt', color='tab:red', linestyle='-')

    # SpinLaunch params
    ax.plot(np.degrees(gamma), v0[np.nanargmin(mass_ratio, axis=1)],
            label='Case 2 opt', color='tab:red', linestyle='--')

    # Minimum params
    ax.plot(np.degrees(gamma), min_v, label='Minimum', color='tab:red', linestyle=':')

    ax.set_xlabel('Launch angle [deg]', fontsize=14)
    ax.set_ylabel('Initial velocity [m/s]', fontsize=14)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True)
    ax.legend(fontsize=12)

    # # plt.savefig('Plots\\Opt_and_min_velocities.pdf')
    plt.show()


    ### Mass diagnostics

    ## Propellant burnt during ascension
    ascension = 900 - (end_prop + circularization_prop)
    ascension[np.where(ascension == 900)] = np.nan
    ascension = fill_nan_2d(ascension)

    plot_2D_figure(ascension, 'Mass burnt in ascension [kg]', extended=True, save='ascension_propellant',
                   plot_paths=True)


    ## Propellant burnt during circularization
    circularization_prop[np.where(circularization_prop <= 0)] = np.nan
    circularization_prop = fill_nan_2d(circularization_prop)

    # Extend the array
    circularization_prop_ext = np.empty((circularization_prop.shape[0] * 2, circularization_prop.shape[1]))
    circularization_prop_ext[::2] = circularization_prop
    circularization_prop_ext[1::2] = circularization_prop

    # Plot results
    plot_2D_figure(circularization_prop, 'Mass burnt in circularization [kg]', extended=True,
                   save='circularization_propellant', contour=True, plot_paths=True)

