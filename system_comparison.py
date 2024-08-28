from OrbitalDynamics.launch_trajectories import moon_launch_optim
import matplotlib.pyplot as plt
import numpy as np


def polynomial(x, *p):
    return sum([p[i] * x ** i for i in range(len(p))])


if __name__ == '__main__':
    # Rocket parameters
    Isp = 300  # s

    # Baseline
    baseline_ratio = np.exp(1.8e3 / (Isp * 9.81)) - 1

    ### Maglev
    deltaV_maglev, launch_params_maglev = moon_launch_optim(return_params=True)
    maglev_ratio = np.exp(deltaV_maglev / (Isp * 9.81)) - 1

    ### SpinLaunch
    gamma = np.radians(np.arange(0.5, 90, 0.5))
    v0 = np.linspace(200, 2000, len(gamma) + 1)

    # Calculate ratio
    # spinLaunch_ratio_0, optimal_v, mp_data = optimize_mass_ratio(gamma, v0, Isp, plot_mass_ratio=False)
    mass_ratio = np.loadtxt('filled_mass_ratio.csv', delimiter=',')
    mass_ratio[mass_ratio <= 0] = np.nan

    # Extend the array horizontally for better plotting
    mass_ratio_T = mass_ratio.T

    mass_ratio_T_duplicated = np.empty((mass_ratio_T.shape[0], mass_ratio_T.shape[1] * 2))
    mass_ratio_T_duplicated[:, ::2] = mass_ratio_T
    mass_ratio_T_duplicated[:, 1::2] = mass_ratio_T

    # Plot spinLaunch
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a heatmap
    cax = ax.imshow(mass_ratio_T_duplicated, aspect='auto', origin='lower', cmap='viridis')

    # Create color bar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mass ratio', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add contour lines
    contours = plt.contour(mass_ratio_T_duplicated, colors='white', linewidths=1)
    plt.clabel(contours, inline=True, fontsize=12, fmt='%1.2f')

    # Create labels
    ax.set_xlabel('Launch angle [deg]', fontsize=14)
    ax.set_ylabel('Initial velocity [m/s]', fontsize=14)

    # Formatting
    x_labels = np.array(np.arange(10, 90, 10), dtype=int)
    x_ticks = np.searchsorted(gamma, np.radians(x_labels)) * 2
    plt.xticks(ticks=x_ticks,
               labels=x_labels,
               fontsize=12)

    y_labels = np.arange(200, 2000, 200)
    y_ticks = np.searchsorted(v0, y_labels)
    plt.yticks(ticks=y_ticks,
               labels=y_labels,
               fontsize=12)

    # Plot optimal path
    min_ratio = np.nanargmin(mass_ratio_T_duplicated, axis=0)
    launch_params_maglev_ext = np.empty((launch_params_maglev.shape[0] * 2, launch_params_maglev.shape[1]))
    launch_params_maglev_ext[::2] = launch_params_maglev
    launch_params_maglev_ext[1::2] = launch_params_maglev

    ax.plot(np.searchsorted(gamma, launch_params_maglev[:, 0]) * 2,
            np.searchsorted(v0, launch_params_maglev[:, 1]),
            color='tab:red', linestyle='--', label='Case 1 opt')
    ax.plot(np.arange(len(min_ratio)), min_ratio, color='tab:red', linestyle='-', label='Case 2 opt')
    ax.grid(True)

    fig.legend(bbox_to_anchor=(0.3, 0.31), fontsize=12)
    fig.tight_layout()
    plt.savefig('Plots\\spinLaunch_mass_ratio.pdf')
    plt.show()

    ## Optimum and minimum velocities and their corresponding ratios
    first_non_nan_indices = np.full(mass_ratio_T.shape[1], np.nan)

    # Iterate over each column to find the first non-NaN value
    ratio_min_v = []
    min_v = []
    for col in range(mass_ratio_T.shape[1]):
        non_nan_indices = np.where(~np.isnan(mass_ratio_T[:, col]))[0]
        if non_nan_indices.size > 0:
            first_non_nan_indices[col] = int(non_nan_indices[0])
            ratio_min_v.append(mass_ratio_T[int(non_nan_indices[0]), col])
            min_v.append(v0[int(non_nan_indices[0])])

    # Plot ratios
    fig, ax = plt.subplots(figsize=(12, 6))

    # Maglev params
    ax.plot(np.degrees(launch_params_maglev[:, 0]), maglev_ratio,
            label=r'Case 1 opt', color='tab:blue', linestyle='-')

    # SpinLaunch params
    ax.plot(np.degrees(gamma), np.nanmin(mass_ratio, axis=1),
            label='Case 2 opt', color='tab:blue', linestyle='--')

    # Minimum params
    ax.plot(np.degrees(gamma), ratio_min_v, label='Minimum', color='tab:blue', linestyle=':')

    ax.set_xlabel('Launch angle [deg]', fontsize=14)
    ax.set_ylabel('Mass ratio [-]', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True)
    ax.legend(bbox_to_anchor=(0.64, 0.945), fontsize=12)

    plt.savefig('Plots\\Opt_and_min_ratios.pdf')
    plt.show()

    # Plot velocities
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

    plt.savefig('Plots\\Opt_and_min_velocities.pdf')
    plt.show()
