from src.physics import PhysicsObject

import toml
import numpy as np
from pathlib import Path
import pandas as pd
import qutip as qt
from itertools import product
import copy
import hashlib


def setup_results_folder(config):
    results_folder = Path(config['meta']['folder'])
    results_folder.mkdir(parents=True, exist_ok=True)
    images_folder = results_folder / 'img'
    images_folder.mkdir(parents=True, exist_ok=True)

    # Save the configuration file
    with open(results_folder / 'config.toml', 'w') as f:
        toml.dump(config, f)

    return results_folder


def read_configuration(args=None):
    """Read the configuration file and generate parameter combinations if lists are detected."""
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    # Overwrite the configuration with the command line arguments
    if args is not None and args.chains:
        config['ancilla']['chains'] = args.chains
    if args is not None and args.layers:
        config['ancilla']['layers'] = args.layers

    # Detect lists in the configuration and generate parameter combinations
    param_keys = []
    param_values = []
    for section, params in config.items():
        for key, value in params.items():
            if isinstance(value, list):
                param_keys.append((section, key))
                param_values.append(value)

    if not param_keys:
        # Single configuration case
        config['id'] = ''
        results_folder = setup_results_folder(config)

        configs = [(config, results_folder)]
    
    # Generate Cartesian product of all parameter combinations
    param_combinations = list(product(*param_values))
    configs = []
    for combination in param_combinations:
        new_config = copy.deepcopy(config)
        for i, (section, key) in enumerate(param_keys):
            new_config[section][key] = combination[i]
            
        # Generate a unique ID for this configuration
        config_id = hashlib.md5(str(combination).encode()).hexdigest()[:8]
        new_config['id'] = config_id

        # Create the folder to store the results for this configuration
        results_folder = setup_results_folder(new_config)
        configs.append((new_config, results_folder))
    return configs


def create_multiindex_dataframe(data):
    """
    QFI data is given in the shape (n, k, t) where t is the number of temperature samplings.
    Create a DataFrame with MultiIndex (t, k).
    """
    n, k, t = data.shape
    # Create MultiIndex
    multi_index = pd.MultiIndex.from_product([range(t), range(k)], 
                                             names=['T', 'A'])
    
    # Reshape data to match MultiIndex
    # Transpose and reshape to (t*k, n)
    reshaped_data = data.transpose(2, 1, 0).reshape(t*k, n)
    
    # Create DataFrame
    return pd.DataFrame(reshaped_data, index=multi_index)




def boltzmann_distribution(omega, T: float):
    """Calculate the Boltzmann distribution for a given frequency and temperature."""
    return 1/(np.exp(omega/T) - 1)




def add_ancilla(rho, ancilla, t, physics: PhysicsObject):
    """
    Add an ancilla to the collective state. If this is not the first ancilla of the layer,
    information from the previous ancilla is gathered via an interaction.
    """
    tspan = np.linspace(0, t, 100)
    subsystems = len(rho.dims[0])
    # Take together the ancilla and the previous correlated state
    rho = qt.tensor(rho, ancilla)
    # Check if the state comprises only the system or other ancillas
    if subsystems > 1:
        # Activate the exchange interaction
        h_exchange = physics.exchange_hamiltonian
        # Extend the Hilbert space to include previous subsystems
        system_dimension = qt.qeye(physics.ndims)
        h_exchange = qt.tensor(system_dimension, *[qt.qeye(2) for _ in range(subsystems-2)], h_exchange)
        rho_evolution = qt.mesolve(
            h_exchange, rho, tspan)
    
        rho = rho_evolution.final_state

    return rho


if __name__ == "__main__":
    configs = read_configuration()