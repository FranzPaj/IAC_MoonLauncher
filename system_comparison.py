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

    # Plot maglev
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Mass ratio
    ax1.plot(np.degrees(launch_params_maglev[:, 0]), maglev_ratio,
             label='Mass Ratio', color='tab:blue')
    ax1.set_xlabel('Launch Angle [deg]', fontsize=14)
    ax1.set_ylabel('Mass ratio [-]', fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # Velocity
    ax2 = ax1.twinx()
    ax2.plot(np.degrees(launch_params_maglev[:, 0]), launch_params_maglev[:, 1],
             label=r'V$_0$', color='tab:red', linestyle='--')
    ax2.set_ylabel('Initial velocity [-]', fontsize=14, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)

    # Formatting
    ax1.grid(True)

    fig.legend(bbox_to_anchor=(0.9, 0.575), fontsize=12)
    fig.tight_layout()
    plt.savefig('Plots\\maglev_mass_ratio_and_v.pdf')
    plt.show()

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
            color='lime', linestyle='dashdot', label='Case 1 opt')
    ax.plot(np.arange(len(min_ratio)), min_ratio, 'r--', label='Case 2 opt')
    ax.grid(True)

    fig.legend(bbox_to_anchor=(0.3, 0.31), fontsize=12)
    fig.tight_layout()
    plt.savefig('Plots\\spinLaunch_mass_ratio.pdf')
    plt.show()

    ## Mass ratio comparisson
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(np.degrees(launch_params_maglev[:, 0]), maglev_ratio,
            color='tab:blue', linestyle='-', label='Case 1')
    ax.plot(np.degrees(gamma), np.nanmin(mass_ratio, axis=1),
            color='tab:orange', linestyle='--', label='Case 2 opt')

    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel('Launch angle [deg]', fontsize=14)
    plt.ylabel('Mass ratio [-]', fontsize=14)

    plt.legend(prop={'size': 12})
    plt.grid(True)

    plt.savefig('Plots\\Opt_mass_ratio_overview.pdf')
    plt.show()
