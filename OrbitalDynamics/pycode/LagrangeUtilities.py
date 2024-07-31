'''
Helper functions aimed at aiding int he study of Circularly Restricted 3 Body Problems (CR3BPs). These functions propagate orbits from an initial
condition or use the continuation technique for obtaining a closed orbit in the vicinity of Lagrange points starting from semi-periodic orbits.
The functions here have already been validated during the Special Topics in Astrodynamics coursework.
'''

# General imports
import numpy as np
import scipy
import warnings

# Custom imports
from pycode.CustomConstants import mu_combined as MU_STD


###########################################################################
# HELPER FUNCTIONS ########################################################
###########################################################################

def fun_3d(t, state, mu, beta, orientation_type, orientation):
    '''
    Helper function for the IVP integrator in 3D space. It receives the 6X1 state vector and returns the state derivatives.
    It can also handle the dynamics of solar sails, but only constant orientation and Sun-collinear control laws are implemented.
    '''

    # Extract state
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # Compute useful quantities
    r = np.array([x, y, z])
    r1 = np.array([x + mu, y, z])
    r1_norm = np.linalg.norm(r1)
    r1_hat = r1 / r1_norm
    r2 = np.array([x - (1 - mu), y, z])
    v = np.array([vx, vy, vz]) 
    if orientation_type == 'constant':
        n_hat = orientation
    # Compute derivatives
    rdot = v
    vdot = - ( (1 - mu) / np.linalg.norm(r1)**3 * r1 + mu / np.linalg.norm(r2)**3 * r2 ) - 2 * np.array([-vy, vx, 0]) + np.array([x, y, 0])+ \
    beta * (1-mu) / np.linalg.norm(r1)**2 * (np.dot(n_hat, r1_hat))**2 * n_hat
    # Find complete state derivative
    state_dot = np.concatenate((rdot,vdot))

    return state_dot

def fun_var(t, state, mu, beta, orientation_type, orientation):
    '''
    Helper function for the IVP integrator in 3D space, including the propagation of the variational equations. 
    It receives the 42X1 statevector and returns the state derivatives. It can also handle the dynamics of solar
    sails, but only constant orientation control laws are implemented.
    '''

    # Extract state
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # Extract State Transition Matrix
    PHI = np.reshape(state[6:], (6,6))
    # Compute useful quantities
    r = np.array([x, y, z])
    r1 = np.array([x + mu, y, z])
    r1_norm = np.linalg.norm(r1)
    r1_hat = r1 / r1_norm
    r2 = np.array([x - (1 - mu), y, z])
    r2_norm = np.linalg.norm(r2)
    v = np.array([vx, vy, vz]) 
    # Define solar sail orientation
    beta_xx = 0
    beta_xy = 0
    beta_xz = 0
    if orientation_type == 'constant':
        n_hat = orientation
        if np.all(orientation == np.array([1, 0, 0])):
            beta_xx = 2 * beta * (1 - mu)/ r1_norm**4 * (x + mu) * (1 - 2 * (x + mu)**2 / r1_norm**2)
            beta_xy = - 4 * beta * (1 - mu) / r1_norm**6 * y * (x + mu)**2
            beta_xz = - 4 * beta * (1 - mu) / r1_norm**6 * z * (x + mu)**2

    # Compute derivatives of state
    rdot = v
    vdot = - ( (1 - mu) / np.linalg.norm(r1)**3 * r1 + mu / np.linalg.norm(r2)**3 * r2 ) - 2 * np.array([-vy, vx, 0]) + np.array([x, y, 0]) + \
    beta * (1 - mu) / r1_norm**2 * (np.dot(n_hat, r1_hat))**2 * n_hat
    # Derivative of STM
    #### Define A 
    A = np.zeros((6,6))
    A[0:3, 3:6] = np.array([[1,0,0], [0,1,0], [0,0,1]])
    A[3:6, 3:6] = np.array([[0,2,0], [-2,0,0], [0,0,0]])
    # Define potential derivatives
    Uxx = -1 + (1 - mu)/ r1_norm**3 * (1 - 3 * (x + mu)**2 / r1_norm**2) + mu / r2_norm**3 * (1 - 3 * (x - (1 - mu))**2 / r2_norm**2)
    Uyy = -1 + (1 - mu)/ r1_norm**3 * (1 - 3 * y**2 / r1_norm**2) + mu / r2_norm**3 * (1 - 3 * y**2 / r2_norm**2)
    Uzz = + (1 - mu)/ r1_norm**3 * (1 - 3 * z**2 / r1_norm**2) + mu / r2_norm**3 * (1 - 3 * z**2 / r2_norm**2)
    Uxy = -3 * (x + mu) * (1 - mu) / r1_norm**5 * y - 3 * (x - (1 - mu)) * mu / r2_norm**5 * y
    Uxz = -3 * (x + mu) * (1 - mu) / r1_norm**5 * z - 3 * (x - (1 - mu)) * mu / r2_norm**5 * z
    Uyz =  z * y * (-3 * (1 - mu) / r1_norm**5 - 3 * mu / r2_norm**5)
    # Give values to matrix
    A [3:6,0:3] = np.array([[-Uxx + beta_xx, -Uxy + beta_xy, -Uxz + beta_xz], [-Uxy, -Uyy, -Uyz], [-Uxz, -Uyz, -Uzz]])
    ####
    PHIdot = np.matmul(A,PHI)
    # Find complete state derivative
    state_dot = np.concatenate((rdot,vdot,PHIdot.flatten()))

    return state_dot

def propagate_3d_orbit(initial_state, t_span, tol, args=(MU_STD, 0, 'constant', np.array([0,0,0]))):
    '''
    Function for propagating a 3D orbit in the CR3BP starting from an initial condition.
    '''

    sol = scipy.integrate.solve_ivp(fun_3d, t_span, initial_state, method='RK45', dense_output = True, rtol = tol, atol = tol, args = args)

    return sol.t, sol.y.T

def orbit_continuation(x0, mu, beta = 0, orientation_type = 'constant', orientation = np.zeros(3), print_flag = False):
    '''
    Function that uses a continuation technique to obtain a periodic orbit from a first approximated semi-periodic solution.
    '''

    print('Continuation started')
    # Define stop condition at mid orbit
    def stop_condition(t, state, mu, beta, orientation_type, orientation):
        return state[1]
    stop_condition.terminal = True

    # Perform continuation
    t_span = [0,4*np.pi]
    tol = 1E-12
    args = (mu, beta, orientation_type, orientation)
    error_norm = 1
    iter = 0
    # Conduct integrations
    while error_norm > 1E-12 and iter < 10:
        # Integrate
        x0_with_stm = np.concatenate((x0, np.eye(6).flatten()))
        sol = scipy.integrate.solve_ivp(fun_var, t_span, x0_with_stm, method='RK45', rtol = tol, atol = tol, args = args, events = stop_condition)
        y_end = sol.y.T[-1,:]
        # print('Initial and final states:', x0,y_end[:6])
        # Isolate final state and final transition matrix
        final_state = y_end[:6]
        final_stm = np.reshape(y_end[6:], (6,6))
        # Identify deviation in xdot and zdot
        xdot_guess = final_state[3]
        zdot_guess = final_state[5]
        error = np.array([xdot_guess, zdot_guess])
        if print_flag:
            print('Error:',error)
        # Create matrix of linear system
        # Find values for STM elements
        phi_10 = final_stm[1,0]
        phi_14 = final_stm[1,4]
        phi_30 = final_stm[3,0]
        phi_34 = final_stm[3,4]
        phi_50 = final_stm[5,0]
        phi_54 = final_stm[5,4]
        # Find values for accelerations
        vdot = compute_total_acceleration(final_state, mu, beta, orientation_type, orientation)
        vx = final_state[3]
        vy = final_state[4]
        vz = final_state[5]
        ax = vdot[0]
        ay = vdot[1]
        az = vdot[2]
        # Set up linear system
        coeff = np.array([0, -xdot_guess, - zdot_guess])
        M = np.array([[phi_10, phi_14, vy], [phi_30, phi_34, ax], [phi_50, phi_54, az]])
        dx, dydot, dt_half = np.linalg.solve(M, coeff)
        # dydot = (zdot * xddot_half - xdot * zddot_half)/(zddot_half * phi_34 - xddot_half * phi_54)
        # Evaluate new initial state
        x0 = x0 + np.array([dx,0,0,0,dydot,0])
        # Define conditions for new cycle

        iter = iter + 1
        error_norm = np.linalg.norm(error)
        x0_sol = x0
    
        if print_flag:
            print('Required correction',dydot, dt_half)
            print('Current error:', error_norm, ', current iteration:', iter)
            print('---*---')
    
    if iter >= 10 and error_norm > 1e-12:
        warnings.warn('WARNING: the differential correct has not converged')
        print('WARNING: the differential correct has not converged!!!')

    return x0_sol, iter

def compute_total_acceleration(state, MU, beta, orientation_type, orientation):
    '''
    Function that computes the total acceleration acting on a spacecraft in the CR3BP.
    '''

    # Extract state
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # Compute useful quantities
    r1 = np.array([x + MU, y, z])
    r2 = np.array([x - (1 - MU), y, z])
    # Define solar sail orientation
    if orientation_type == 'constant':
        n_hat = orientation
    # Compute derivatives of state
    vdot = - ( (1 - MU) / np.linalg.norm(r1)**3 * r1 + MU / np.linalg.norm(r2)**3 * r2 ) - 2 * np.array([-vy, vx, 0]) + np.array([x, y, 0]) + \
    beta * (1-MU)/np.linalg.norm(r1)**3 * (np.dot(n_hat, r1))**2 * n_hat

    return vdot

