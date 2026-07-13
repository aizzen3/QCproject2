# ==================================================
# Imports
# ==================================================

import os
import numpy as np
import matplotlib.pyplot as plt

import quimb.tensor as qtn

from qiskit.quantum_info import Statevector
from qiskit_mps_initializer.datatypes import QuantumState


# ==================================================
# Save folder
# ==================================================

save_dir = "mps_dagger_chi_quimb_results_10layers"
os.makedirs(save_dir, exist_ok=True)

print("Plots will be saved in:", os.path.abspath(save_dir))


# ==================================================
# Settings
# ==================================================

n_qubits = 8
Nx = 2 ** n_qubits

# Run only 10 layers
layer_list = range(0, 11)

# Plot statevector only for layers 1 to 7
statevector_layers_to_plot = range(0, 8)

quimb_cutoff = 1e-3

T = 10e-6
lam = 630e-9
n_refr = 1.0

d0 = 0.0
h = 200e-9
sigma = 1e-6
x0 = 0.0


# ==================================================
# Helper functions
# ==================================================

def save_and_show(filename):
    path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.show()
    plt.close()


def spatial_grid(T, Nx):
    return np.linspace(-T / 2, T / 2, Nx, endpoint=False)


def gaussian_thickness_profile(x, d0, h, sigma, x0=0.0):
    return d0 + h * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def phase_signal(d, n, lam):
    k0 = 2 * np.pi / lam
    return n * k0 * d


def phi_from_f(f):
    f = np.asarray(f, dtype=float)

    if np.any(f < 0):
        raise ValueError("f must be non-negative.")

    alpha = np.sum(f)

    if alpha <= 0:
        raise ValueError("alpha=sum(f) must be positive.")

    psi = np.sqrt(f / alpha)
    psi = psi.astype(complex)
    psi = psi / np.linalg.norm(psi)

    return psi, alpha


def dense_state_to_quimb_mps(psi, n_qubits, cutoff=1e-3, max_bond=None):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)

    psi_tensor = psi.reshape([2] * n_qubits)

    qmps = qtn.MatrixProductState.from_dense(
        psi_tensor,
        dims=[2] * n_qubits,
        cutoff=cutoff,
        max_bond=max_bond,
    )

    return qmps


def quimb_bond_report(qmps):
    bond_sizes = []

    for i in range(qmps.L - 1):
        bond = qmps.bond(i, i + 1)
        bond_sizes.append(qmps.ind_size(bond))

    max_bond = max(bond_sizes) if len(bond_sizes) > 0 else 1

    return bond_sizes, max_bond


# ==================================================
# Build target psi
# ==================================================

print("\nBuilding Gaussian target state")

x = spatial_grid(T, Nx)

d = gaussian_thickness_profile(
    x=x,
    d0=d0,
    h=h,
    sigma=sigma,
    x0=x0,
)

f = phase_signal(
    d=d,
    n=n_refr,
    lam=lam,
)

psi, alpha = phi_from_f(f)

psi = np.asarray(psi, dtype=complex)
psi = psi / np.linalg.norm(psi)

n_qubits = int(np.log2(len(psi)))
Nx = len(psi)

target_state = Statevector(psi)

zero_state = np.zeros_like(psi, dtype=complex)
zero_state[0] = 1.0

print("alpha:", alpha)
print("n_qubits:", n_qubits)
print("Nx:", Nx)
print("norm psi:", np.linalg.norm(psi))
print("min |psi|:", np.min(np.abs(psi)))
print("max |psi|:", np.max(np.abs(psi)))


# ==================================================
# Original psi bond dimension using Quimb
# ==================================================

qmps_psi = dense_state_to_quimb_mps(
    psi=psi,
    n_qubits=n_qubits,
    cutoff=quimb_cutoff,
)

psi_bond_profile, psi_max_bond = quimb_bond_report(qmps_psi)

print("\nOriginal psi MPS bond dimensions")
print("Bond profile:", psi_bond_profile)
print("Max bond dimension:", psi_max_bond)


# ==================================================
# Main loop
# ==================================================

layers = []
max_bond_chi_results = []
bond_profile_chi_results = []
fidelity_zero_results = []
chi_statevector_results = {}

for number_of_layers in layer_list:

    print("\n" + "=" * 70)
    print("number_of_layers:", number_of_layers)
    print("=" * 70)

    # ----------------------------------------------
    # Generate MPS initializer circuit U_psi(L)
    # ----------------------------------------------

    psi_state = QuantumState.from_dense_data(
        data=psi,
        normalize=True,
    )

    U_psi = psi_state.generate_mps_initializer_circuit(
        number_of_layers=number_of_layers,
    )

    # ----------------------------------------------
    # Dagger
    # ----------------------------------------------

    U_psi_dagger = U_psi.inverse()

    # ----------------------------------------------
    # chi_L = U_psi_dagger(L) |psi>
    # ----------------------------------------------

    chi_state = target_state.evolve(U_psi_dagger)

    chi = np.asarray(chi_state.data, dtype=complex)
    chi = chi / np.linalg.norm(chi)

    # ----------------------------------------------
    # Quimb MPS bond dimension of chi_L
    # ----------------------------------------------

    qmps_chi = dense_state_to_quimb_mps(
        psi=chi,
        n_qubits=n_qubits,
        cutoff=quimb_cutoff,
    )

    bond_profile_chi, max_bond_chi = quimb_bond_report(qmps_chi)

    # ----------------------------------------------
    # Fidelity with |000...0>
    # ----------------------------------------------

    fidelity_zero = np.abs(np.vdot(zero_state, chi)) ** 2
    fidelity_zero = float(np.real(fidelity_zero))

    # ----------------------------------------------
    # Store
    # ----------------------------------------------

    layers.append(number_of_layers)
    max_bond_chi_results.append(max_bond_chi)
    bond_profile_chi_results.append(bond_profile_chi)
    fidelity_zero_results.append(fidelity_zero)

    if number_of_layers in statevector_layers_to_plot:
        chi_statevector_results[number_of_layers] = chi

    # ----------------------------------------------
    # Print
    # ----------------------------------------------

    print("Bond profile of chi:", bond_profile_chi)
    print("Max bond dimension of chi:", max_bond_chi)
    print("Fidelity chi with |000...0>:", fidelity_zero)
    print("Circuit depth:", U_psi.depth())
    print("Circuit size:", U_psi.size())


# ==================================================
# Convert results to arrays
# ==================================================

layers = np.array(layers)
max_bond_chi_results = np.array(max_bond_chi_results)
fidelity_zero_results = np.array(fidelity_zero_results)


# ==================================================
# Save numerical results
# ==================================================

results_table = np.column_stack(
    [
        layers,
        max_bond_chi_results,
        fidelity_zero_results,
    ]
)

results_path = os.path.join(save_dir, "chi_layer_bond_fidelity_results.csv")

np.savetxt(
    results_path,
    results_table,
    delimiter=",",
    header="layer,max_bond_chi,fidelity_with_zero",
    comments="",
)

print("\nSaved CSV:", results_path)


# ==================================================
# Plot 1: Max bond dimension vs layers
# ==================================================

plt.figure(figsize=(7, 5))

plt.plot(
    layers,
    max_bond_chi_results,
    marker="o",
    linewidth=1,
)

plt.xlabel("Number of MPS initializer layers")
plt.ylabel("Max bond dimension of chi")
plt.title(f"Max bond dimension of chi vs layers, {n_qubits} qubits")
plt.grid(True)

save_and_show(f"chi_max_bond_vs_layers_{n_qubits}_qubits.png")


# ==================================================
# Plot 2: Fidelity with |000...0> vs layers
# ==================================================

plt.figure(figsize=(7, 5))

plt.plot(
    layers,
    fidelity_zero_results,
    marker="o",
    linewidth=1,
)

plt.xlabel("Number of MPS initializer layers")
plt.ylabel("Fidelity with |000...0>")
plt.title(f"Fidelity of chi with zero state vs layers, {n_qubits} qubits")
plt.grid(True)

save_and_show(f"chi_fidelity_zero_vs_layers_{n_qubits}_qubits.png")


# ==================================================
# Plot 3: Statevector amplitude plots for layers 1 to 7
# ==================================================

basis_index = np.arange(Nx)

for layer, chi_plot in chi_statevector_results.items():

    plt.figure(figsize=(8, 5))

    plt.plot(
        basis_index,
        np.abs(chi_plot),
        marker="o",
        markersize=3,
        linewidth=1,
    )

    plt.xlabel("Computational basis index")
    plt.ylabel("|chi amplitude|")
    plt.title(
        f"Statevector amplitude |chi| after U dagger\n"
        f"Layer {layer}, {n_qubits} qubits"
    )
    plt.grid(True)

    save_and_show(
        f"chi_statevector_amplitude_layer_{layer}_{n_qubits}_qubits.png"
    )


# ==================================================
# Final summary
# ==================================================

print("\nDone.")
print("All plots saved in:", os.path.abspath(save_dir))
print("CSV saved as:", results_path)