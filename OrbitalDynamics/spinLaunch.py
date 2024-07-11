import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycode.HelperFunctions import Rocket, gravitational_param_dict, average_radius_dict


class SpinLaunch:
    # TODO: Implement parameter optimization in class
    def __init__(self,
                 initial_velocity: float,
                 initial_angle: float,
                 thruster: Rocket,
                 thrust_load: None or float = None):

        self.v = initial_velocity
        self.gamma = initial_angle
        self.thruster = thruster
        self.thrust_load = thrust_load

    def gravity_turn(self):
        # Initial conditions
        t = 0
        dt = 0.01
        g = 1.625
        pos = np.array([0, 0])
        v = self.v
        gamma = self.gamma

        # Initialize the lists to store the values
        time = [t]
        velocity = [v]
        angle = [gamma]
        mass = [self.thruster.mass]
        thrust_history = []

        thrust = self.thruster.thrust
        while gamma >= 0:
            # Calculate the thrust
            if self.thrust_load is not None:
                thrust = self.thrust_load * self.thruster.mass * g

            if self.thruster.mp <= 0:
                thrust = 0

            dV = thrust - g * np.sin(gamma)
            dG = 1 / v * -g * np.cos(gamma)

            # Update the state
            gamma += dG * dt
            v += dV * dt
            t += dt

            # Update the mass
            if self.thruster.mp > 0:
                self.thruster.impulse(dV*dt)

            # Store values
            pos = np.vstack((pos, pos[-1] + v * dt * np.array([np.cos(gamma), np.sin(gamma)])))
            velocity.append(v)
            angle.append(gamma)
            time.append(t)
            mass.append(self.thruster.mass)
            thrust_history.append(thrust)



        return pos, np.array(velocity), np.array(angle), np.array(time), np.array(mass), np.array(thrust_history)

    def plot_launch(self, pos):
        plt.plot(pos[:, 0], pos[:, 1])
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title('Launch trajectory')
        plt.show()


def launch(angle, v0,
           Isp: float = 300,
           construction: float = 0.1,
           mass: float = 1000,
           thrust: float = 1000):

    g = 1.625
    v_target = np.sqrt(gravitational_param_dict['Moon'] / average_radius_dict['Moon'] + 1e5)

    dt = 1e-2
    prop_mass_ratio = 1 / 10

    convergence = False
    falling = False
    removed_mass = False

    while not convergence:
        total_mass = mass
        mp = total_mass * (1 - construction) * prop_mass_ratio  # Propellant to be burnt during ascent

        t = 0
        v = v0
        gamma = angle
        pos = np.array([0, 0])

        # Start simulation
        while gamma > 0:
            # Calculate thrust
            if mp <= 0:
                thrust = 0

            # EOM
            dV = thrust - g * np.sin(gamma)
            dG = 1 / v * -g * np.cos(gamma)

            # Update the state
            gamma += dG * dt
            v += dV * dt
            t += dt

            # Update mass
            dm = total_mass * (1 - np.exp(-dV * dt / (Isp * 9.81)))
            total_mass -= dm
            mp -= dm

            # Store position values and check end conditions
            pos = np.vstack((pos, pos[-1] + v * dt * np.array([np.cos(gamma), np.sin(gamma)])))

            if pos[-1, 1] < pos[-2, 1]:  # We are falling
                falling = True
                break

            if pos[-1, 1] > 1e5:
                break

        # Analyze exit conditions
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
            if gamma <= np.radians(5) and pos[-1, 1] >= 1e5:
                convergence = True
            else:
                if pos[-1, 1] >= 1e5:
                    # Rocket could have burnt less mass
                    prop_mass_ratio -= 1 / 20
                    removed_mass = True
                    if prop_mass_ratio <= 0:
                        prop_mass_ratio += 1 / 20
                        convergence = True
                else:
                    prop_mass_ratio += 1 / 20
                    if removed_mass:
                        # The solution lies between the previous and current mass fractions
                        convergence = True

    deltaV = np.sqrt(v ** 2 + v_target ** 2 - 2 * v * v_target * np.cos(gamma))
    dm = total_mass * (1 - np.exp(-deltaV / (Isp * 9.81)))

    # Return propellant mass fraction
    return (1 - construction) * prop_mass_ratio + dm / mass


def optimize_initial_params():
    # Define optimization parameters
    gamma = np.radians(np.arange(0, 90, 5))
    v0 = np.linspace(0, 1000, len(gamma))

    # Initialize arrays to save solution
    prop_mass_fraction = np.zeros((len(gamma) - 1, len(v0) - 1))

    # Start simulation
    for i, angle in tqdm(enumerate(gamma[1:])):
        for j, vel in enumerate(v0[1:]):
            prop_mass_fraction[i, j] = launch(angle, vel)

    return prop_mass_fraction


if __name__ == '__main__':
    gamma = np.radians(np.arange(0, 90, 5))
    v0 = np.linspace(0, 1000, len(gamma))

    prop_mass = optimize_initial_params()

    plt.imshow(prop_mass)

    cbar = plt.colorbar()
    cbar.set_label('Propellant mass consumed [kg]')

    plt.xticks(ticks=np.arange(0, len(v0[1:]), 2), labels=np.array(np.round(v0[1::2]), dtype=int))
    plt.yticks(ticks=np.arange(0, len(gamma[1:]), 5), labels=np.array(np.round(np.degrees(gamma[1::5])), dtype=int))

    plt.xlabel('Initial velocity [m/s]')
    plt.ylabel('Launch angle')
    plt.title('Propellant mass consumed for a 1000kg rocket with Isp = 300s')

    plt.show()

