"""
Functions to calculate Quantum Fisher Information.

___________
References
[1] Seah, S., Nimmrichter, S., Grimmer, D., Santos, J. P., Scarani, V., & Landi, G. T. (2019). Collisional quantum thermometry. Physical review letters, 123(18), 180602.
"""

from itertools import product
import qutip


def thermal_fisher_information(ancilla, T):
    """
    Compute the thermal Fisher Information that bounds our QFI by the Cramér-Rao inequality.
    This depends only on the measurement system (ancillas) and the temperature.
    See [1].
    """
    c = ancilla.heat_capacity(T)
    return c / T**2


def compute_fisher_information(rho: qutip.Qobj, dr: qutip.Qobj):
    """Compute the Quantum Fisher Information."""

    ndims = rho.shape[0]
    qfi = 0
    for n, m in product(range(ndims), range(ndims)):
        # Compute expectation values
        psi_n = qutip.basis(ndims, n)
        psi_m = qutip.basis(ndims, m)
        rho_n = qutip.expect(rho, psi_n)
        rho_m = qutip.expect(rho, psi_m)
        dr_nm = qutip.expect(dr, psi_n * psi_m.dag())
        # Use the formula (13) from the paper
        if rho_n + rho_m != 0:
            qfi += 2 * dr_nm**2 / (rho_n + rho_m)

    return qfi