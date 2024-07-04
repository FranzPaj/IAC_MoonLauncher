import numpy as np
import matplotlib.pyplot as plt
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

            if pos[-1, 1] < pos[-2, 1]:  # We are falling
                print('Max height = ', pos[-2, 1])
                pos = pos[:-1]
                break

            if pos[-1, 1] > 1e5:
                print('We reached 100km')
                print('Remaining propellant mass = ', self.thruster.mp)
                print('Current velocity = ', v)
                break

        return pos, np.array(velocity), np.array(angle), np.array(time), np.array(mass), np.array(thrust_history)

    def plot_launch(self, pos):
        plt.plot(pos[:, 0], pos[:, 1])
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.title('Launch trajectory')
        plt.show()



if __name__ == '__main__':
    thruster = Rocket(specific_impulse=300,
                      thrust=1000,
                      propellant_mass=80,
                      construction_mass=100,
                      useful_mass=920)

    launch = SpinLaunch(initial_velocity=1000,
                        initial_angle=np.radians(30),
                        thruster=thruster,
                        thrust_load=None)

    pos, v, gamma, t, mass, thrust = launch.gravity_turn()
    launch.plot_launch(pos)

    v_target = np.sqrt(gravitational_param_dict['Moon'] / average_radius_dict['Moon'] + 1e5)

    deltaV = np.sqrt(v[-1]**2 + v_target**2 - 2*v[-1]*v_target*np.cos(gamma[-1]))
    print('Delta-V to circularize = ', deltaV)
    print('Propellant mass needed = ', launch.thruster.mass * (1 - np.exp(-deltaV / (9.81 * launch.thruster.Isp))))