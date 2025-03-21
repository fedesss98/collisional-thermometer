"""
Script to calculate the Quantum Fisher Information,
using a cross-correlated collisional framework of measurements.

___________
References
[1] Seah, S., Nimmrichter, S., Grimmer, D., Santos, J. P., Scarani, V., & Landi, G. T. (2019). Collisional quantum thermometry. Physical review letters, 123(18), 180602.
"""

from src.utils import read_configuration
from src.physics import PhysicsObject

import matplotlib.pyplot as plt
import numpy as np
import qutip
from pathlib import Path
from tqdm import tqdm

import argparse

HPLANCK = 1.0
KBOLTZMANN = 1.0

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Quantum Information Measures for a given quantum state.')
    parser.add_argument('--chains', '-k', type=str, help='Number of interacting ancilla chains.')
    parser.add_argument('--depth', '-n', type=int, help='Number of ancillas in each chain.')

    return parser.parse_args()


def initialize_physics():
    """Initialize the physical parameters of the system."""
    return {
        'omega_S': 1.0,  # System frequency
        
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


def boltzmann_distribution(omega, T: float):
    """Calculate the Boltzmann distribution for a given frequency and temperature."""
    return 1/(np.exp(omega/T) - 1)


def thermalize_system(system, temp, t, physics: PhysicsObject):
    """Thermalize the system with the environment."""
    tspan = np.linspace(0, t, 100)
    n_th = boltzmann_distribution(physics.system.frequency, temp)
    jump_operators = physics.jump_operators(n_th)

    rho_evolution = qutip.mesolve(
        physics.system.hamiltonian, system, tspan, jump_operators)
    
    return rho_evolution.final_state


def add_ancilla(rho, ancilla, t, physics: PhysicsObject):
    """
    Add an ancilla to the collective state. If this is not the first ancilla of the layer,
    information from the previous ancilla is gathered via an interaction.
    """
    tspan = np.linspace(0, t, 100)
    subsystems = len(rho.dims[0])
    # Take together the ancilla and the previous correlated state
    rho = qutip.tensor(rho, ancilla)
    # Check if the state comprises only the system or other ancillas
    if subsystems > 1:
        # Activate the exchange interaction
        h_exchange = physics.exchange_hamiltonian
        # Extend the Hilbert space to include previous subsystems
        system_dimension = qutip.qeye(physics.ndims)
        h_exchange = qutip.tensor(system_dimension, *[qutip.qeye(2) for _ in range(subsystems-2)], h_exchange)
        rho_evolution = qutip.mesolve(
            h_exchange, rho, tspan)
    
        rho = rho_evolution.final_state

    return rho


def collision_evolution(rho, t, physics: PhysicsObject):
    """Evolve the collective state following a System-Ancilla collision"""
    tspan = np.linspace(0, t, 100)
    subsystems = len(rho.dims[0])
    # Extend the Hilbert space of the system to include all ancillas
    h_s = physics.extend_hs(subsystems)
    h_a = physics.extend_ha(subsystems)
    h_int = physics.interaction_hamiltonian(subsystems)
    rho_evolution = qutip.mesolve(
        h_s + h_a + h_int, rho, tspan)
    
    return rho_evolution.final_state


def measure_system(rho, ancillas, collision_time, p, pbar: tqdm) -> qutip.Qobj:
    """
    Get information on the System via Ancilla measurements.
    The System collide with each Ancilla in each layer 
    and each ancilla collide with the next one.
    """    
    for j, ancilla in enumerate(ancillas):
        # Now the Hilbert space get bigger and bigger
        # A new Ancilla is added and it gather information from the previous Ancilla
        if j == 0:
            pbar.set_description(f"Composing System with first Ancilla")
        else:
            pbar.set_description(f"Passing information to Ancilla {j+1}")
        
        rho = add_ancilla(rho, ancilla, collision_time, p)
        # The Ancilla is now part of the collective state
        # Evolve the collective state with the collision Hamiltonian
        pbar.set_description(f"Collision with Ancilla {j+1}")
        rho = collision_evolution(rho, collision_time, p)

    # Trace out Ancillas
    system = rho.ptrace(0)

    return system


def thermal_fisher_information(ancilla, T):
    """
    Compute the thermal Fisher Information that bounds our QFI by the Cramér-Rao inequality.
    This depends only on the measurement system (ancillas) and the temperature.
    See [1]
    """
    c = ancilla.heat_capacity(T)
    return c / T**2


def compute_fisher_information(dS, dT, rho):
    """Compute the Quantum Fisher Information."""
    # Compute the derivative of the System state with respect to the temperature
    dr = qutip.Qobj(dS / dT)
    rho = qutip.Qobj(rho)

    # Compute the Quantum Fisher Information
    ndims = rho.shape[0]
    qfi = []
    for n, m in zip(range(ndims), range(ndims)):
        # Compute expectation values
        psi_n = qutip.basis(ndims, n)
        psi_m = qutip.basis(ndims, m)
        rho_n = qutip.expect(rho, psi_n)
        rho_m = qutip.expect(rho, psi_m)
        dr_nm = qutip.expect(dr, psi_n * psi_m.dag())
        # Use the formula (13) from the paper
        if rho_n + rho_m != 0:
            qfi.append(2 * dr_nm**2 / (rho_n + rho_m))

    return sum(qfi)


def plot_qfi(temperature_range, qfi_values, chains, n, folder=None, compare=None):
    """Plot the Quantum Fisher Information."""
    fig, ax = plt.subplots(figsize=(8, 6), layout='tight')
    ax.plot(temperature_range, qfi_values, linewidth=1, marker='o', markersize=4, label="QFI")
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('QFI')
    ax.set_title('Quantum Fisher Information')

    ax.text(0.8, 0.9, f"k = {chains} / n = {n}", transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, pad=10))
    
    if compare is not None:
        # Add a reference series to compare with
        ax.plot(temperature_range, compare, 'r--', linewidth=1, label="Reference QFI")
        fig.legend()

    # Save the plot
    if folder is not None:
        fig.savefig(folder / 'img/qfi.png')
        fig.savefig(folder / 'img/qfi.eps')

    plt.show()



def main(args):
    # Parse the arguments
    config, root = read_configuration(args)

    p = PhysicsObject(config)

    # Create the range of Temperatures to scan
    temperature_range = np.linspace(
        config['thermometer']['T_min'], 
        config['thermometer']['T_max'], 
        config['thermometer']['accuracy'])

    # Average thermalization time for each possible temperature
    thermalization_time = config['system']['thermalization_time']
    # Interactions with the Ancillas
    chains = config['ancilla']['chains']
    iterations = config['ancilla']['layers']
    collision_time = config['ancilla']['collision_time']

    # Create the system and the first layer of ancillas
    system, ancillas = create_systems(p, config['ancilla']['chains'])

    systems = []
    pbar = tqdm(temperature_range)  # Progress bar to show the code running

    for i, temperature in enumerate(pbar):
        pbar.set_description(f"Temperature: {temperature:.2f} K")

        pbar_child = tqdm(range(iterations), leave=False)
        for n in pbar_child:
            system = thermalize_system(system, temperature, thermalization_time, p)
            # Use the collisional framework to measure the System
            system = measure_system(system, ancillas, collision_time, p, pbar_child)

        # Save the System state after the Ancilla measurements
        systems.append(system.full())  # Save the state as a numpy array
        
        # Close the progress bar
        pbar_child.close()
    
    pbar.close()

    # Compute the change in the state with respect to the change in temperature
    system_variations = np.diff(systems, axis=0)
    temperature_variations = np.diff(temperature_range)

    # Calculate the Quantum Fisher Information for every temperature change
    qfi_values = []
    for i, dT in enumerate(temperature_variations):
        dS = system_variations[i]
        qfi = compute_fisher_information(dS, dT, systems[i+1])
        qfi_values.append(qfi)
    
    # Calculate the Thermal Fisher Information given by the Cramer-Rao inequality
    tfi_values = [thermal_fisher_information(p.ancilla, T) for T in temperature_range[1:]]

    # Plot the Quantum Fisher Information
    plot_qfi(temperature_range[1:], qfi_values, chains, n, root, compare=tfi_values)





if __name__ == '__main__':
    args = parse_arguments()
    main(args)
