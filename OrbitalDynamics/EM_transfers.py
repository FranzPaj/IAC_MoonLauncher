from pycode.HelperFunctions import Orbit, gravitational_param_dict, average_radius_dict, sma_moon
import numpy as np



if __name__ == '__main__':
    # Earth Moon 2D transfer
    # TODO: Implement polar transfer

    # Define initial parameters
    h_origin = 100e3  # m
    body_origin = 'Moon'

    v_moon = np.sqrt(gravitational_param_dict['Earth'] / sma_moon)
    SOI_moon = 66100e3  # m

    mu0 = gravitational_param_dict[body_origin]
    R0 = average_radius_dict[body_origin]

    # Define target parameters
    h_target = 300e3  # m
    body_target = 'Earth'
    muE = gravitational_param_dict[body_target]
    Re = average_radius_dict[body_target]

    # Define transfer orbit
    sma_transfer = (sma_moon + (R0 + h_origin) + (Re + h_target)) / 2
    e_transfer = 1 - (Re + h_target) / sma_transfer

    transfer_orbit = Orbit('Earth', sma_transfer, e_transfer, inclination=0)

    # Calculate parameters at the border of the SOI
    theta_SOI = np.arccos((transfer_orbit.p / transfer_orbit.sma - 1) / e_transfer)
    r_SOI = np.sqrt(sma_moon**2 + SOI_moon**2 - 2 * sma_moon * SOI_moon * np.cos(np.pi - theta_SOI))
    v_SOI = np.sqrt(2 * (muE / r_SOI - muE / (2 * sma_transfer)))
    gamma_SOI = np.arccos(transfer_orbit.h / (r_SOI * v_SOI))

    # Moon escape manoeuvre
    v_inf = v_SOI * np.array([[np.cos(gamma_SOI)], [np.sin(gamma_SOI)]]) - np.array([[0], [v_moon]])
    v_inf = np.linalg.norm(v_inf)

    v0 = np.sqrt(mu0 / (R0 + h_origin))
    deltaV = np.sqrt(2 * (v_inf**2 / 2 + mu0 / (R0 + h_origin))) - v0

    print(f'Moon escape manoeuvre: {deltaV} m/s')

