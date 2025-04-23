"""
Script to calculate the Quantum Fisher Information,
using a cross-correlated collisional framework of measurements [1].

It generates simulations based on the config.toml file located in the root directory.
If the config file presents lists of parameters, the script iterates through all of them,
creating unique ids and folders to differentiate them.
Some parameter can be overridden via command line arguments.

Results are saved in the directory specified in the config file, and in the case of multiple parameters,
in subdirectories identified by the ids of each parameter iteration.
___________
References
[1] Mendonça, T. M., Soares-Pinto, D. O., & Paternostro, M. (2024). Information flow-enhanced precision in collisional quantum thermometry. arXiv preprint arXiv:2407.21618.
[2] Seah, S., Nimmrichter, S., Grimmer, D., Santos, J. P., Scarani, V., & Landi, G. T. (2019). Collisional quantum thermometry. Physical review letters, 123(18), 180602.
"""

from src.utils import read_configuration, create_multiindex_dataframe, look_for_incomplete
from src.physics import PhysicsObject
from src.time_evolutions import thermalize_system, measure_system
from src.quantum_fisher_information import compute_fisher_information

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip
from tqdm import tqdm

import pickle

import argparse

HPLANCK = 1.0
KBOLTZMANN = 1.0

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Quantum Information Measures for a given quantum state.')
    parser.add_argument('--chains', '-k', type=str, help='Number of interacting ancilla chains.')
    parser.add_argument('--layers', '-n', type=int, help='Number of ancillas in each chain.')
    parser.add_argument('--plot', '-p', action='store_true', help='Show the comparison of QFI from the last ancilla of every chain (default: Flase).')
    parser.add_argument('--resume-file', '-r', default=None, type=str, help='Resume analysis for configurations without results given a common root.')

    return parser.parse_args()


def gather_configs(args=None):
    if args is not None and args.resume_file is not None:
        # Look for folder in the root directory with no qfi results
        configs = look_for_incomplete(args.resume_file)
    else:
        # Read configuration and parse the arguments in it
        configs = read_configuration(args)

    return configs



def create_systems(physics: PhysicsObject, k):
    """Create the system and ancilla states."""
    # Create the system
    system = physics.system.dm
    # system = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    # Create the first layer of ancillas across the chains
    ancilla_dm = physics.ancilla.dm
    # ancilla_dm = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    ancillas = [ancilla_dm for _ in range(k)]

    return system, ancillas


def iterate_qfi_computations(states, dt):
    state_variations = np.diff(states, axis=0)  # Compute finite differential
    variables = zip(states, state_variations)
    return np.array([compute_fisher_information(state, ds, dt) for state, ds in variables])


def _calculate_qfi(systems, ancillas, tmin, tmax, samples, layers, chains, results_folder):

    # Calculate the Quantum Fisher Information for every temperature change and for every ancilla
    dT = (tmax - tmin) / samples
    temperatures = np.linspace(tmin, tmax, samples)
    # Go through all the saved states to compute Quantum Fisher Information
    qfi_system = {}
    qfi_ancillas = {}
    system_evolution = np.array(list(systems.values()))
    ancillas_evolution = np.array(list(ancillas.values()))
    for n in range(layers):
        systems = system_evolution[:, n]
        derivatives = np.diff(systems, axis=0) / dT
        qfi_system[n] = [compute_fisher_information(s, ds) for s, ds in zip(systems, derivatives)]
        ancillas = ancillas_evolution[:, n]
        derivatives = np.diff(ancillas, axis=0) / dT
        qfi_ancillas[n] = [
            [compute_fisher_information(a, da) for a, da in zip(ancillas[:, k], derivatives[:, k])] 
            for k in range(chains)]
    
    # Save Quantun Fisher Information in Pickle format
    qfi_system = pd.DataFrame(qfi_system)
    # qfi_system.to_pickle(results_folder / "qfi_systems.pickle")
    qfi_ancillas = create_multiindex_dataframe(
        np.array(list(qfi_ancillas.values())), temperatures)
    # qfi_ancillas.to_pickle(results_folder / "qfi_ancillas.pickle")

    return qfi_ancillas, qfi_system


def save_last_qfi(qfi_df, results_folder):
    """Save the QFI of the last ancilla"""
    last_ancilla = qfi_df.index.get_level_values(1).max()
    last_layer = qfi_df.columns.max()
    last_ancilla_qfi = qfi_df.loc[(slice(None), last_ancilla), last_layer]
    # Drop Ancilla index
    last_ancilla_qfi = last_ancilla_qfi.droplevel(1)
    last_ancilla_qfi.to_pickle(results_folder / "last_ancilla_qfi.pickle")


def plot_qfi(df, chains, n, show=False, folder=None, compare=None):
    """Plot the Quantum Fisher Information for the last layer of ancillas."""
    idx = pd.IndexSlice
    # Take the last layer of the chain
    data_to_plot = df.iloc[:, -1]
    x = data_to_plot.index.get_level_values(0).unique()  # Temperatures
    ys = [data_to_plot.loc[idx[:, i],].values for i in range(chains)]

    fig, ax = plt.subplots(figsize=(8, 6), layout='tight')

    for k in range(chains):
        ax.plot(x, ys[k], linewidth=1, marker='o', markersize=4, label=f"k={k+1}")
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('QFI')
    ax.set_title('Quantum Fisher Information')
    fig.legend()

    ax.text(0.8, 0.9, f"k = {chains} / n = {n}", transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='lightgrey', pad=10))
    
    if compare is not None:
        # Add a reference series to compare with
        ax.plot(x, compare, 'r--', linewidth=1, label="Reference QFI")
    
    # Save the plot
    if folder is not None:
        fig.savefig(folder / 'img/qfi.png')
        fig.savefig(folder / 'img/qfi.eps')

    # Don't show the plot to let the code running
    if show:
        plt.show()
    
    return None


def main(args=None):
    configs = gather_configs(args)
    print(f"Starting Analysis of {len(configs)} configurations")

    # Iterate over all sets of configuration parameters
    for config, results_folder in configs:
        p = PhysicsObject(config)
        # Draw and save circuit representation of the model
        p.circuit.draw_circuit(aspect=(1, 1.0), usetex=True, show=args.plot)
        p.circuit.save_circuit(f"{results_folder}/img/circuit")

        # Create the range of Temperatures to scan
        tmin = config['thermometer']['T_min']
        tmax = config['thermometer']['T_max']
        samples = config['thermometer']['accuracy']
        temperature_range = np.linspace(tmin, tmax, samples)

        # Average thermalization time for each possible temperature
        thermalization_time = config['system']['thermalization_time']
        # Interactions with the Ancillas
        chains = config['ancilla']['chains']
        iterations = config['ancilla']['layers']
        collision_time = config['ancilla']['collision_time']
        exchange_time = config['ancilla']['intra_interaction_time']
        
        print(f"Running simulation with {chains} chains of {iterations} ancillas")
        
        # Create the system and the first layer of ancillas
        system, ancillas = create_systems(p, config['ancilla']['chains'])

        system_evolution = {}
        ancillas_evolution = {}
        pbar = tqdm(temperature_range)  # Progress bar to show the code running

        for i, temperature in enumerate(pbar):
            pbar.set_description(f"Temperature: {temperature:.2f} K")

            system_evolution[temperature] = []
            ancillas_evolution[temperature] = []
 
            pbar_child = tqdm(range(iterations), leave=False)
            for n in pbar_child:
                system = thermalize_system(system, temperature, thermalization_time, p)
                # Use the collisional framework to measure the System
                system, layer_ancillas = measure_system(
                    system, ancillas, exchange_time, collision_time, p, pbar_child)

                # Save the System state after the Ancilla measurements
                system_evolution[temperature].append(system)  # Save the state as a numpy array
                ancillas_evolution[temperature].append(layer_ancillas)

            # Close the progress bar
            pbar_child.close()
        
        pbar.close()

        # Save evolution of the states
        qutip.qsave(system_evolution, f"{results_folder}/system_evolution")
        qutip.qsave(ancillas_evolution, f"{results_folder}/ancillas_evolution")

        qfi_ancillas, qfi_system = _calculate_qfi(
            system_evolution, ancillas_evolution,
            tmin, tmax, samples, iterations, chains, results_folder)
        # Save last ancilla QFI
        save_last_qfi(qfi_ancillas, results_folder)
        # Standard plot of QFI vs T for the last layer of ancillas
        plot_qfi(
            qfi_ancillas, chains, iterations, show=args.plot, folder=results_folder)
        plt.close('all')
        print(f"Done!\nFile saved in {results_folder}")


def main_old(args=None):
    # Read configuration and parse the arguments in it
    configs = read_configuration(args)

    # Iterate over all sets of configuration parameters
    for config, results_folder in configs:
        p = PhysicsObject(config)
        # Draw and save circuit representation of the model
        p.circuit.draw_circuit(aspect=(1, 1.0), usetex=True, show=args.plot)
        p.circuit.save_circuit(f"{results_folder}/img/circuit")

        # Create the range of Temperatures to scan
        tmin = config['thermometer']['T_min']
        tmax = config['thermometer']['T_max']
        samples = config['thermometer']['accuracy']
        temperature_range = np.linspace(tmin, tmax, samples)

        # Average thermalization time for each possible temperature
        thermalization_time = config['system']['thermalization_time']
        # Interactions with the Ancillas
        chains = config['ancilla']['chains']
        iterations = config['ancilla']['layers']
        collision_time = config['ancilla']['collision_time']
        exchange_time = config['ancilla']['intra_interaction_time']
        
        print(f"Running simulation with {chains} chains of {iterations} ancillas")
        
        # Create the system and the first layer of ancillas
        system, ancillas = create_systems(p, config['ancilla']['chains'])

        system_evolution = {}
        ancillas_evolution = {}
        pbar = tqdm(temperature_range)  # Progress bar to show the code running

        for i, temperature in enumerate(pbar):
            pbar.set_description(f"Temperature: {temperature:.2f} K")

            system_evolution[temperature] = []
            ancillas_evolution[temperature] = []
 
            pbar_child = tqdm(range(iterations), leave=False)
            for n in pbar_child:
                system = thermalize_system(system, temperature, thermalization_time, p)
                # Use the collisional framework to measure the System
                system, layer_ancillas = measure_system(
                    system, ancillas, exchange_time, collision_time, p, pbar_child)

                # Save the System state after the Ancilla measurements
                system_evolution[temperature].append(system)  # Save the state as a numpy array
                ancillas_evolution[temperature].append(layer_ancillas)

            # Close the progress bar
            pbar_child.close()
        
        pbar.close()

        # Save evolution of the states
        qutip.qsave(system_evolution, f"{results_folder}/system_evolution")
        qutip.qsave(ancillas_evolution, f"{results_folder}/ancillas_evolution")

        qfi_ancillas, qfi_system = _calculate_qfi(
            system_evolution, ancillas_evolution,
            tmin, tmax, samples, iterations, chains, results_folder)
        # Save last ancilla QFI
        save_last_qfi(qfi_ancillas, results_folder)
        # Standard plot of QFI vs T for the last layer of ancillas
        plot_qfi(
            qfi_ancillas, chains, iterations, show=args.plot, folder=results_folder)
        plt.close('all')
        print(f"Done!\nFile saved in {results_folder}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
