import numpy as np
import unittest

from src.physics import PhysicsObject
from src.time_evolutions import thermalize_system, measure_system
from tests.main import create_systems, fast_config

def test_maximum_qfi(args):
    p, t_range, system, ancillas, times = args
    thermalization_t, collision_t = times

    system_evolution = {[] for _ in t_range}
    ancillas_evolution = {[] for _ in t_range}

    for i, temperature in enumerate(p, t_range):

        system = thermalize_system(system, temperature, thermalization_t, p)
        # Use the collisional framework to measure the System
        system, layer_ancillas = measure_system(system, ancillas, collision_t, p)

        # Save the System state after the Ancilla measurements
        system_evolution[temperature].append(system)  # Save the state as a numpy array
        ancillas_evolution[temperature].append(layer_ancillas)

        

def main():
    config = fast_config()
    p = PhysicsObject(config)
    chains = 3
    thermalization_time = np.pi/2
    collision_time = np.pi/2

    temperature_range = range(
        config["thermometer"]["T_min"], config["thermometer"]["T_max"], 
        config["thermometer"]["accuracy"])
    # Create the system and the first layer of ancillas
    system, ancillas = create_systems(p, chains)

    args = [
        p, temperature_range, system, ancillas,
        (thermalization_time, collision_time)
    ]

    test_maximum_qfi(*args)

if __name__ == "__main__":
    main()