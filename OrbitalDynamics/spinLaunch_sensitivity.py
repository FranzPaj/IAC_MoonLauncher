import numpy as np
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.analyze import sobol
from spinLaunch import launch


# Define the problem
problem = {
    'num_vars': 6,
    'names': ['angle', 'v0', 'Isp','construction', 'mass', 'thrust'],
    'bounds': [[0.05, np.pi / 2 - 0.05],
               [0, 1000],
               [200, 400],
               [0.05, 0.15],
               [200, 1200],
               [500, 1500]]
}

# Sample values
param_values = saltelli.sample(problem, 2**6)

# Evaluate the model
mass_fractions = np.array([launch(*params) for params in tqdm(param_values)])

print(f'Total sample size = {len(mass_fractions)}')

# Filter out the NaN values
param_values = param_values[~np.isnan(mass_fractions)]
mass_fractions = mass_fractions[~np.isnan(mass_fractions)]

print(f'Filtered sample size = {len(mass_fractions)}')

while len(mass_fractions) % (2 * problem['num_vars'] + 2) != 0:
    mass_fractions = mass_fractions[:-1]
    param_values = param_values[:-1]

print('Reduced sample size =', len(mass_fractions))

# Analyze results
sobol_indices = sobol.analyze(problem, mass_fractions, print_to_console=True)

S1s = np.array([s['S1'] for s in sobol_indices])  # First order indices
STs = np.array([s['ST'] for s in sobol_indices])  # Total indices

print('First-order sobol indices:')
print(S1s, '\n')

print('Total sobol indices:')
print(STs)

