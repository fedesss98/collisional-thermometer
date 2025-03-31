
from src.utils import read_configuration, create_multiindex_dataframe
from src.physics import PhysicsObject
from src.time_evolutions import thermalize_system, measure_system
from src.quantum_fisher_information import compute_fisher_information

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip
from tqdm import tqdm

import argparse

HPLANCK = 1.0
KBOLTZMANN = 1.0


def fast_config():
    meta = {"folder": "tests/results", "description": "Test"}
    system = {
        "type": "Qubit",
        "ndims": 2,
        "excitations": 1,
        "frequency": 1.0,
        "thermalization_time": 100
    }
    environment = {
        "g": 1.0,
        "J": 1.0,
        "gamma": 1.0,
    }
    ancilla = {
        "type": "Qubit",
        "ndims": 2,
        "excitations": 1,
        "chains": 3,
        "layers": 1,
        "frequency": 1.0,
        "g": 3.0,
        "intra_interaction_time": 1.570796326,
        "collision_time": 0.031415926,
    }
    thermometer = {
        "T_min": 0.002,
        "T_max": 2.0,
        "accuracy": 100,
    }
    
    return {
        "meta": meta,
        "system": system,
        "environment": environment,
        "ancilla": ancilla,
        "thermometer": thermometer,
    }


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
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, pad=10))
    
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
