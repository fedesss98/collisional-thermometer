"""
Physics module, this stores all the physics related functions and classes
"""
import numpy as np
import qutip as qt

class PhysicsObject:
    def __init__(self, config):
        """Initialize variables given the configuration dictionary."""
        self.system_type = config['system']['type']
        self.ndims = config['system']['ndims']
        self.system = config['system']
        self.ancilla = config['ancilla']

        self.environment_coupling = config['environment']['g']
        self.ancillas_coupling = config['ancilla']['g']

        self.gamma_d = config['environment']['gamma']  # Dissipation rate    

    @property
    def system(self):
        return self._system
    
    @property
    def ancilla(self):
        return self._ancilla

    @system.setter
    def system(self, config):
        excitations = config['excitations']
        ndims = config['ndims']
        frequency = config['frequency']

        if self.system_type == "Qubit":
            self._system = Qubit(ndims, excitations, frequency, "Qubit")
        elif self.system_type in ["Cavity", "Field"]:
            self._system = Field(ndims, excitations, frequency, "Field")
        elif self.system_type == "Phaseonium":
            coherences = config['coherences']
            self._system = Phaseonium(ndims, excitations, frequency, coherences, "Phaseonium")
        else:
            raise ValueError(f"Cannot use System of type {self.system_type}")
        
    @ancilla.setter
    def ancilla(self, config):
        excitations = config['excitations']
        ndims = config['ndims']
        frequency = config['frequency']

        if self.system_type == "Qubit":
            self._ancilla = Qubit(ndims, excitations, frequency, "Qubit")
        elif self.system_type in ["Cavity", "Field"]:
            self._ancilla = Field(ndims, excitations, frequency, "Field")
        elif self.system_type == "Phaseonium":
            coherences = config['coherences']
            self._ancilla = Phaseonium(ndims, excitations, frequency, coherences, "Phaseonium")
        else:
            raise ValueError(f"Cannot use System of type {self.system_type}")

    def jump_operators(self, n_th):
        """Return the thermal jump operators for the system"""
        jump_up = np.sqrt(self.gamma_d * (1 + n_th)) * self.system.create_operator 
        jump_down = np.sqrt(self.gamma_d * (n_th)) * self.system.destroy_operator
        
        return [jump_up, jump_down]
    
    @property
    def exchange_hamiltonian(self):
        """Return the exchange Hamiltonian for two Ancillas"""
        h1 = qt.tensor(self.ancilla.hamiltonian, qt.qeye(2))
        h2 = qt.tensor(qt.qeye(2), self.ancilla.hamiltonian)
        h12 = self.ancillas_coupling * (
            qt.tensor(self.ancilla.destroy_operator, self.ancilla.create_operator) + 
                qt.tensor(self.ancilla.create_operator, self.ancilla.destroy_operator)
        )
        return h1 + h2 + h12
    
    def extend_hs(self, subsystems):
        """Extend the Hilbert space of the System to include all ancillas"""
        extended_dimension = [qt.qeye(2) for _ in range(subsystems - 1)]
        return qt.tensor(self.system.hamiltonian, *extended_dimension)
    
    def extend_ha(self, subsystems):
        """Extend the Hilbert space of the Ancilla to include all previous subsystems"""
        system_dimension = qt.qeye(self.ndims)
        extended_dimension = [qt.qeye(2) for _ in range(subsystems - 2)]
        return qt.tensor(
            system_dimension, *extended_dimension, self.ancilla.hamiltonian
        )
    
    def interaction_hamiltonian(self, subsystems):
        """Return the interaction Hamiltonian between the System and the Ancilla"""
        jump_up = qt.tensor(
            self.system.create_operator,
            *[qt.qeye(2) for _ in range(subsystems - 2)],
            self.ancilla.destroy_operator)
        jump_down = qt.tensor(
            self.system.destroy_operator, 
            *[qt.qeye(2) for _ in range(subsystems - 2)],
            self.ancilla.create_operator)
        return self.ancillas_coupling * (jump_up + jump_down)

   

class System:
    def __init__(self, ndims, excitations, frequency, name) -> None:
        self.ndims = ndims
        self.name = name
        self.frequency = frequency
        self.excitations = excitations
        self.state = qt.basis(ndims, excitations)

    @property
    def hamiltonian(self):
        """Return the Hamiltonian of the System in its subspace"""
        pass

    @property
    def jump_up_operator(self):
        """Return the Operator that increase System energy"""
        pass

    @property
    def jump_down_operator(self):
        """Return the Operator that decrease System energy"""
        pass

    @property
    def dm(self):
        """Return the Density Matrix of the System"""
        return qt.ket2dm(self.state)
    
    def heat_capacity(self, temp):
        """
        Return the Heat Capacity of the System, defined as:
        <H^2> - <H>^2 / kT^2
        """
        variance = qt.variance(self.hamiltonian, self.state)
        return variance / (temp ** 2)


class Qubit(System):
    def __init__(self, ndims, excitations, frequency, name) -> None:
        super().__init__(ndims, excitations, frequency, name)
        # State for qubits are inverted, 0 is |up> and 1 is |down>
        self.state = qt.basis(2, 1 - excitations)

    @property
    def hamiltonian(self):
        return 0.5 * self.frequency * qt.sigmaz()

    @property
    def create_operator(self):
        return qt.sigmap()
    
    @property
    def destroy_operator(self):
        return qt.sigmam()


class Field(System):
    def __init__(self, ndims, excitations, frequency, name) -> None:
        super().__init__(ndims, excitations, frequency, name)
        # Override default system density matrix
        self.state = qt.basis(ndims, excitations)

    @property
    def hamiltonian(self):
        """In the number state basis the Hamiltonian reads hbar*omega*(N+1/2)"""
        op = qt.num(self.ndims) + 0.5 * qt.qeye(self.ndims)
        return self.frequency * op

    @property
    def create_operator(self):
        return qt.create(self.ndims)

    @property
    def destroy_operator(self):
        return qt.destroy(self.ndims)
    

class Phaseonium(System):
    def __init__(self, ndims, excitations, frequency, coherences, name) -> None:
        super().__init__(ndims, excitations, frequency, name)
        # The state of phaseoniums have the same notation of qubits
        self.state = qt.basis(3, 1 - excitations)
        self.coherences = coherences

    @property
    def hamiltonian(self):
        matrix = qt.Qobj([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        return self.frequency * qt.num(3)
    
    @property
    def create_operator(self):
        return qt.Qobj([[0, 1, 1], [0, 0, 0], [0, 0, 0]])

    @property
    def destroy_operator(self):
        return 0.5 * qt.Qobj([[0, 0, 0], [1, 0, 0], [1, 0, 0]])

    @property
    def dm(self):
        # Add coherences to the density matrix
        state = qt.ket2dm(self.state).full()
        state[1, 2] = self.coherences
        state[2, 1] = np.conjugate(self.coherences)
        return qt.Qobj(state)

