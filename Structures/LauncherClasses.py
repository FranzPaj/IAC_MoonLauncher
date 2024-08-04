import numpy as np


class Launcher:
    def __init__(self, launch_velocity, launch_angle, max_power, useful_mass):
        self.vel = launch_velocity
        self.angle = launch_angle
        self.max_power = max_power
        self.mass = useful_mass


class MaglevLauncher(Launcher):
    def __init__(self, launch_velocity, launch_angle, max_power, max_acceleration):
        super().__init__(launch_velocity, launch_angle, max_power)
        self.max_acceleration = max_acceleration

    def constant_power_launch(self):
        """
        Function for calculating launch parameters for a scenario with constant power.
        Power = Force * velocity --> Neglecting launcher mass --> Power = mass * acceleration * velocity
        :returns: energy, distance, time
        """

        # Initiate variables at the point where the acceleration needs to be reduced
        velocity = self.max_power / (self.mass * self.max_acceleration)
        time = velocity / self.max_acceleration
        distance = 0.5 * self.max_acceleration * time ** 2

        # Propagate until launch
        dt = 1e-2
        while velocity < self.vel:
            acceleration = self.max_power / (self.mass * velocity)
            velocity += acceleration * dt
            distance += velocity * dt
            time += dt

        # Calculate tota energy consumption
        energy = self.max_power * time
        return energy, distance, time

    def constant_acceleration_launch(self):
        """
        Function for calculating launch parameters for a scenario with constant acceleration.
        :returns: energy, distance, time
        """

        time = self.vel / self.max_acceleration
        distance = 0.5 * self.max_acceleration * time ** 2
        energy = self.mass * self.vel ** 2 / 2
        return energy, distance, time
        
