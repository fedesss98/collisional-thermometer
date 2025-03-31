from src.utils import boltzmann_distribution, add_ancilla
from src.physics import PhysicsObject

import numpy as np
from tqdm import tqdm
import qutip


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


def measure_system(rho, ancillas, collision_time, p, pbar: tqdm = None) -> qutip.Qobj:
    """
    Get information on the System via Ancilla measurements.
    The System collide with each Ancilla in each layer 
    and each ancilla collide with the next one.
    """    
    for j, ancilla in enumerate(ancillas):
        # Now the Hilbert space get bigger and bigger
        # A new Ancilla is added and it gather information from the previous Ancilla
        if j == 0:
            pbar.set_description("Composing System with first Ancilla")
        else:
            pbar.set_description(f"Passing information to Ancilla {j+1}")
        
        rho = add_ancilla(rho, ancilla, collision_time, p)
        # The Ancilla is now part of the collective state
        # Evolve the collective state with the collision Hamiltonian
        if pbar is not None:
            # Update the Progress Bar
            pbar.set_description(f"Collision with Ancilla {j+1}")
        rho = collision_evolution(rho, collision_time, p)

    # Separate subsystems
    system = rho.ptrace(0)
    layer_ancillas = [rho.ptrace(i+1) for i in range(len(ancillas))]

    return system, layer_ancillas



def thermalize_system(system, temp, t, physics: PhysicsObject):
    """Thermalize the system with the environment."""
    tspan = np.linspace(0, t, 100)
    n_th = boltzmann_distribution(physics.system.frequency, temp)
    jump_operators = physics.jump_operators(n_th)

    rho_evolution = qutip.mesolve(
        physics.system.hamiltonian, system, tspan, jump_operators)
    
    return rho_evolution.final_state


