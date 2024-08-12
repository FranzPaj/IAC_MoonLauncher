from OrbitalDynamics.pycode.HelperFunctions import Transfer, Orbit
from OrbitalDynamics.pycode.CustomConstants import average_radius_dict
import matplotlib.pyplot as plt
import numpy as np


def calculate_fuel_mass(rocket_data: dict, transfer: Transfer, deltaV_override: np.ndarray = None, payload_density=50):
    """
    Calculate the fuel mass of a rocket given the data of the rocket assuming oxidizer refueling is possible.
    :param rocket_data: dict with the following:
        - empty_mass: float, empty mass of the rocket [kg]
        - propellant_mass: float, mass of the propellant [kg]
        - OF_ratio: float, oxidizer to fuel ratio [-]
    :return: float, mass of the fuel
    """

    delta_v = transfer.get_transfer_delta_V() if deltaV_override is None else deltaV_override
    mass_ratio = np.exp(delta_v / (rocket_data['Isp'] * 9.81))
    payload_mass = payload_density * rocket_data['payload_V']
    prop_mass = (rocket_data['propellant_mass'] + rocket_data['empty_mass'] + payload_mass) * (1 - 1 / mass_ratio)

    return rocket_data['propellant_mass'] - prop_mass


# Falcon 9 v1.2 and Falcon heavy's data - second stage
# Retrieved from https://sma.nasa.gov/LaunchVehicle/assets/spacex-falcon-9-v1.2-data-sheet.pdf
Falcon_v12 = {
    'empty_mass': 4.5e3,  # kg
    'propellant_mass': 111.5e3,  # kg
    'thrust': 934e3,  # N
    'Isp': 348,  # s
    'burn_time': 397,  # s
    'OF_ratio': 2.56,  # oxidizer to fuel ratio
    'payload_V': 16 * np.pi * 1.83**2  # m^3
}

Falcon_Heavy = {  # From https://www.wevolver.com/specs/falcon-heavy-block-5
    'empty_mass': 22200,  # kg
    'propellant_mass': 433100 - 22200,  # kg
    'thrust': 24681e3,  # N - With boosters
    'Isp': 311,  # s
    'burn_time': 187,  # s
    'OF_ratio': 2.56,  # oxidizer to fuel ratio
}

if __name__ == '__main__':
    # dV = np.linspace(0, 10e3, 1000)
    # fuel_mass = calculate_fuel_mass(Falcon_v12, None, dV)
    # ox_mass = Falcon_v12['OF_ratio'] * fuel_mass
    #
    # prop_mass_ratio = (fuel_mass + ox_mass) / Falcon_v12['propellant_mass']
    # # prop_mass_ratio[np.where(prop_mass_ratio > 1)] = 1
    #
    # full_refueling = dV[np.where(prop_mass_ratio < 1)[0][0]]
    #
    # no_refueling = np.where(prop_mass_ratio <= 0)[0][0]
    # dV = dV[:no_refueling]
    # prop_mass_ratio = prop_mass_ratio[:no_refueling]
    #
    # # Plot results
    # plt.plot(dV / 1e3, prop_mass_ratio * 100)
    # plt.vlines(full_refueling / 1e3, 0, 100, color='black', linestyle='--',
    #            label='100% refueling')
    #
    # plt.title('Payload density = 50 kg/m3')
    # plt.xlabel('Delta-V [km/s]')
    # plt.ylabel('Refueling capacity [%]')
    # plt.legend()
    # plt.show()

    # parking orbit altitude needed to be able to refuel completely
    Moon_parking = Orbit('Moon', 100e3 + average_radius_dict['Moon'], 0)

    refueling = 0
    h = 300e3
    while refueling < 1:
        h += 10e3
        Earth_parking = Orbit('Earth', h + average_radius_dict['Earth'], 0)
        direct_transfer = Transfer(Earth_parking, Moon_parking, 'Earth')

        fuel_mass = calculate_fuel_mass(Falcon_v12, direct_transfer)
        ox_mass = Falcon_v12['OF_ratio'] * fuel_mass

        prop_mass = fuel_mass + ox_mass
        refueling = prop_mass/Falcon_v12['propellant_mass']

    print(h)
