import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycode.HelperFunctions import Rocket, gravitational_param_dict, average_radius_dict


class SpinLaunch:
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


def optimize_initial_params(thrust_load: float = None):
    g = 1.625
    v_target = np.sqrt(gravitational_param_dict['Moon'] / average_radius_dict['Moon'] + 1e5)

    # Define engine parameters
    Isp = 300
    construction = 100

    # Define optimization parameters
    gamma = np.radians(np.arange(0, 90, 5))
    v0 = np.linspace(0, 1000, len(gamma))

    # Initialize arrays to save solution
    prop_mass = np.zeros((len(gamma) - 1, len(v0) - 1))

    # Start simulation
    for i in tqdm(range(len(gamma[1:]))):
        # print('\nLaunch angle: ', np.degrees(gamma[i + 1]))
        G = gamma[i + 1]
        for j, vel in enumerate(v0[1:]):
            # print('\n\tInitial velocity: ', vel)

            # Define timestep
            dt = 1e-2

            # Initialize the lists to store the values
            convergence = False
            falling = False
            removed_mass = False
            prop_mass_ratio = 1/9

            while not convergence:
                total_mass = 1000
                mp = (total_mass - construction) * prop_mass_ratio  # Propellant to be burnt during ascent

                t = 0
                v = vel
                angle = G
                pos = np.array([0, 0])

                # Start simulation
                while angle > 0:
                    # Calculate thrust
                    if mp <= 0:
                        thrust = 0
                    else:
                        if thrust_load is None:
                            thrust = 1000
                        else:
                            thrust = thrust_load * total_mass * g

                    # EOM
                    dV = thrust - g * np.sin(angle)
                    dG = 1 / v * -g * np.cos(angle)

                    # Update the state
                    angle += dG * dt
                    v += dV * dt
                    t += dt

                    # Update mass
                    dm = total_mass * (1 - np.exp(-dV * dt / (Isp * 9.81)))
                    total_mass -= dm
                    mp -= dm

                    # Store position values and check end conditions
                    pos = np.vstack((pos, pos[-1] + v * dt * np.array([np.cos(angle), np.sin(angle)])))

                    if pos[-1, 1] < pos[-2, 1]:  # We are falling
                        falling = True
                        break

                    if pos[-1, 1] > 1e5:
                        break

                if falling:
                    if removed_mass:
                        convergence = True
                    else:
                        prop_mass_ratio += 1/18  # Add 50 kg to the propellant mass
                        # print('\t\tAdding mass: ', prop_mass_ratio * 900)
                        if prop_mass_ratio >= 1:
                            # print('\t\tDid not converge')
                            prop_mass_ratio = np.nan
                            convergence = True

                else:
                    if angle <= np.radians(5) and pos[-1, 1] >= 1e5:
                        convergence = True
                        # print('\t\tConverged')
                    else:
                        if pos[-1, 1] >= 1e5:
                            prop_mass_ratio -= 1/18  # Remove 50 kg from the propellant mass
                            # print('\t\tRemoving mass: ', prop_mass_ratio * 900)
                            removed_mass = True
                            if prop_mass_ratio <= 0:
                                prop_mass_ratio += 1 / 18
                                # print('\t\tMinimum mass reached')
                                convergence = True
                        else:
                            if removed_mass:
                                convergence = True
                            else:
                                prop_mass_ratio += 1/18
                                # print('\t\tAdding mass: ', prop_mass_ratio * 900)

            deltaV = np.sqrt(v**2 + v_target**2 - 2*v*v_target*np.cos(angle))
            dm = total_mass * (1 - np.exp(-deltaV / (Isp * 9.81)))
            prop_mass[i, j] = dm + 900 * prop_mass_ratio

    return prop_mass



if __name__ == '__main__':
    # thruster = Rocket(specific_impulse=300,
    #                   thrust=1000,
    #                   propellant_mass=50,
    #                   construction_mass=100,
    #                   useful_mass=850)
    #
    # launch = SpinLaunch(initial_velocity=125,
    #                     initial_angle=np.radians(10),
    #                     thruster=thruster,
    #                     thrust_load=None)
    #
    # pos, v, gamma, t, mass, thrust = launch.gravity_turn()
    # launch.plot_launch(pos)
    # # print('Max altitude = ', pos[-1, 1])
    #
    # v_target = np.sqrt(gravitational_param_dict['Moon'] / average_radius_dict['Moon'] + 1e5)
    #
    # deltaV = np.sqrt(v[-1]**2 + v_target**2 - 2*v[-1]*v_target*np.cos(gamma[-1]))
    # # print('Delta-V to circularize = ', deltaV)
    # # print('Propellant mass needed = ', launch.thruster.mass * (1 - np.exp(-deltaV / (9.81 * launch.thruster.Isp))))

    prop_mass = optimize_initial_params()

    plt.imshow(prop_mass)
    plt.colorbar()
    plt.xlabel('Initial velocity [m/s]')
    plt.ylabel('Launch angle')
    plt.show()
