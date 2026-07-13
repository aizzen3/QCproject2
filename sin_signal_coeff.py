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

save_dir = "sin_quimb_schmidt_target_chi_9q"
os.makedirs(save_dir, exist_ok=True)

print("Plots will be saved in:", os.path.abspath(save_dir))


# ==================================================
# Settings
# ==================================================

n_qubits = 9
Nx = 2 ** n_qubits

layer_list = range(0, 21)   # only 1 to 10 layers

quimb_cutoff = 1e-16
plot_floor = 1e-18

T = 16e-6          # spatial window
lam = 630e-9       # wavelength
n_refr = 1.0

h = 1         # signal height

sinc_scale = 4e6

d0 = 0.0


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


import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_mps_initializer.datatypes import QuantumState
from qiskit.quantum_info import state_fidelity


# -----------------------------
# Spatial grid: fully symmetric
# -----------------------------
def spatial_grid(L, Nx):
   

    dx = L / Nx
    j = np.arange(Nx)

    return (j - (Nx - 1) / 2) * dx


# -----------------------------
# sinc thickness profile: sin(x)/x
# -----------------------------
def thickness_profile_sinc(x, h, scale=1.0, x0=0.0):


    z = scale * (x - x0)

    sinc = np.ones_like(z)
    nonzero = np.abs(z) > 1e-14

    sinc[nonzero] = np.sin(z[nonzero]) / z[nonzero]

    return h * sinc


# -----------------------------
# Positive sinc thickness profile
# -----------------------------
def thickness_profile_positive_sinc(x, h, scale=1.0, x0=0.0):


    z = scale * (x - x0)

    sinc = np.ones_like(z)
    nonzero = np.abs(z) > 1e-14

    sinc[nonzero] = np.sin(z[nonzero]) / z[nonzero]

    # Shift upward so minimum becomes zero
    sinc_positive = sinc - np.min(sinc)

    # Normalize between 0 and 1
    max_val = np.max(sinc_positive)

    if max_val > 0:
        sinc_positive = sinc_positive / max_val

    return h * sinc_positive


# -----------------------------
# Convert thickness to phase signal
# -----------------------------
def phase_signal(d, n_refr, lam):
    k0 = 2 * np.pi / lam
    return n_refr * k0 * d


# -----------------------------
# Convert phase signal to quantum state amplitudes
# -----------------------------
def phi_from_f(f):
    alpha = np.sum(f)

    if alpha <= 0:
        raise ValueError(
            "alpha is zero or negative. Your signal cannot be converted "
            "to amplitudes using sqrt(f / alpha)."
        )

    if np.any(f < 0):
        raise ValueError(
            "f contains negative values. Use thickness_profile_positive_sinc "
            "instead of raw thickness_profile_sinc."
        )

    phi = np.sqrt(f / alpha)

    return phi, alpha


def dense_state_to_quimb_mps(psi, n_qubits, cutoff=1e-16, max_bond=None):
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


def quimb_schmidt_values(qmps, cut, cutoff=1e-16):
    
   
    


    qmps = qmps.copy()
    

    # Quimb returns the Schmidt values across the bond.
    s = qmps.schmidt_values(cut)

    s = np.asarray(s, dtype=float)

    # remove tiny numerical noise
    s[np.abs(s) < cutoff] = 0.0

    # keep descending order
    s = np.sort(s)[::-1]

    return s


def pad_to_length(arr, length):
    arr = np.asarray(arr, dtype=float)
    k = min(len(arr), length)
    out = np.zeros(length, dtype=float)
    out[:k] = arr[:k]
    return out



# Build target psi


print("\nBuilding Gaussian target state")

x = spatial_grid(T, Nx)

d = thickness_profile_positive_sinc(x=x, h=h, scale=sinc_scale, x0=0.0)

f = phase_signal(
    d=d,
    n_refr=n_refr,
    lam=lam,
)

psi, alpha = phi_from_f(f)

psi = np.asarray(psi, dtype=complex)
psi = psi / np.linalg.norm(psi)

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
# Plot original signal
# ==================================================

plt.figure(figsize=(7, 5))

plt.plot(
    x * 1e6,
    f,
    marker="o",
    markersize=3,
    linewidth=1,
)

plt.xlabel("x (µm)")
plt.ylabel("f(x)")
plt.title(f"Gaussian phase signal, {n_qubits} qubits")
plt.grid(True)

save_and_show(f"gaussian_signal_{n_qubits}_qubits.png")


# ==================================================
# Convert target psi to Quimb MPS
# ==================================================

qmps_psi = dense_state_to_quimb_mps(
    psi=psi,
    n_qubits=n_qubits,
    cutoff=quimb_cutoff,
    max_bond=None,
)

psi_bond_profile, psi_max_bond = quimb_bond_report(qmps_psi)

print("\nTarget psi Quimb MPS")
print("Bond profile:", psi_bond_profile)
print("Max bond dimension:", psi_max_bond)



# Cuts



cuts_to_plot = range(1, n_qubits)




# ==================================================
# Target Schmidt coefficients from Quimb
# ==================================================

target_schmidt_by_cut = {}

print("\nTarget Schmidt coefficients from Quimb")

for cut in cuts_to_plot:
    s_target = quimb_schmidt_values(
        qmps=qmps_psi,
        cut=cut,
        cutoff=quimb_cutoff,
    )

    target_schmidt_by_cut[cut] = s_target

    print("\nCut:", cut)
    print("Target Schmidt values:", s_target)
    print("Nonzero count:", np.sum(s_target > 0))


# ==================================================
# Storage
# ==================================================

layers = []
fidelity_zero_results = []
circuit_depth_results = []
circuit_size_results = []

chi_schmidt_by_cut = {cut: [] for cut in cuts_to_plot}
chi_bond_profile_results = []
chi_max_bond_results = []


# ==================================================
# Main loop over number_of_layers
# ==================================================

for number_of_layers in layer_list:

    print("\n" + "=" * 70)
    print("number_of_layers:", number_of_layers)
    print("=" * 70)

    # ----------------------------------------------
    # Build MPS initializer circuit U_psi(L)
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
    # chi_L = U_psi(L)^dagger |psi>
    # ----------------------------------------------

    chi_state = target_state.evolve(U_psi_dagger)

    chi = np.asarray(chi_state.data, dtype=complex)
    chi = chi / np.linalg.norm(chi)

    # ----------------------------------------------
    # Convert chi_L to Quimb MPS
    # ----------------------------------------------

    qmps_chi = dense_state_to_quimb_mps(
        psi=chi,
        n_qubits=n_qubits,
        cutoff=quimb_cutoff,
        max_bond=None,
    )

    chi_bond_profile, chi_max_bond = quimb_bond_report(qmps_chi)

  

    fidelity_zero = np.abs(np.vdot(zero_state, chi)) ** 2
    fidelity_zero = float(np.real(fidelity_zero))

    
    # Quimb Schmidt values for each cut
    

    for cut in cuts_to_plot:
        s_chi = quimb_schmidt_values(
            qmps=qmps_chi,
            cut=cut,
            cutoff=quimb_cutoff,
        )

        chi_schmidt_by_cut[cut].append(s_chi)

    
    # Store
    

    layers.append(number_of_layers)
    fidelity_zero_results.append(fidelity_zero)
    circuit_depth_results.append(U_psi.depth())
    circuit_size_results.append(U_psi.size())
    chi_bond_profile_results.append(chi_bond_profile)
    chi_max_bond_results.append(chi_max_bond)

    # ----------------------------------------------
    # Print
    # ----------------------------------------------

    print("Fidelity chi with |000...0>:", fidelity_zero)
    print("Chi bond profile:", chi_bond_profile)
    print("Chi max bond dimension:", chi_max_bond)
    print("Circuit depth:", U_psi.depth())
    print("Circuit size:", U_psi.size())

    for cut in cuts_to_plot:
        s_chi = chi_schmidt_by_cut[cut][-1]
        nonzero = np.sum(s_chi > 0)

        min_nonzero = np.min(s_chi[s_chi > 0]) if nonzero > 0 else 0.0

        print(
            f"Cut {cut}:",
            "Schmidt count =",
            nonzero,
            "| largest =",
            np.max(s_chi),
            "| smallest nonzero =",
            min_nonzero,
        )


# ==================================================
# Convert results
# ==================================================

layers = np.array(layers)
fidelity_zero_results = np.array(fidelity_zero_results)
circuit_depth_results = np.array(circuit_depth_results)
circuit_size_results = np.array(circuit_size_results)
chi_max_bond_results = np.array(chi_max_bond_results)


# ==================================================
# Save summary CSV
# ==================================================

summary_table = np.column_stack(
    [
        layers,
        fidelity_zero_results,
        chi_max_bond_results,
        circuit_depth_results,
        circuit_size_results,
    ]
)

summary_path = os.path.join(save_dir, "summary_layers.csv")

np.savetxt(
    summary_path,
    summary_table,
    delimiter=",",
    header="layer,fidelity_with_zero,chi_max_bond,circuit_depth,circuit_size",
    comments="",
)

print("\nSaved summary CSV:", summary_path)


# ==================================================
# Plot fidelity
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
plt.title(f"Fidelity of chi with zero state, {n_qubits} qubits")
plt.grid(True)

save_and_show(f"chi_fidelity_zero_vs_layers_{n_qubits}_qubits.png")


# ==================================================
# Plot max bond dimension
# ==================================================

plt.figure(figsize=(7, 5))

plt.plot(
    layers,
    chi_max_bond_results,
    marker="o",
    linewidth=1,
)

plt.xlabel("Number of MPS initializer layers")
plt.ylabel("Max bond dimension of chi")
plt.title(f"Quimb max bond dimension of chi, {n_qubits} qubits")
plt.grid(True)

save_and_show(f"chi_max_bond_vs_layers_{n_qubits}_qubits.png")


# ==================================================
# Schmidt coefficient plots
#
# One plot per cut.
# Solid lines = chi Schmidt coefficients vs layers.
# Dashed horizontal lines = target Schmidt coefficients.
# ==================================================

for cut in cuts_to_plot:

    print("\nPlotting Schmidt coefficients for cut:", cut)

    max_rank = min(2 ** cut, 2 ** (n_qubits - cut))

    s_target = pad_to_length(target_schmidt_by_cut[cut], max_rank)

    chi_matrix = []

    for s_chi in chi_schmidt_by_cut[cut]:
        chi_matrix.append(pad_to_length(s_chi, max_rank))

    chi_matrix = np.array(chi_matrix)

    # ----------------------------------------------
    # Save CSV for this cut
    # ----------------------------------------------

    target_repeated = np.tile(s_target, (len(layers), 1))

    table = np.column_stack(
        [
            layers,
            chi_matrix,
            target_repeated,
        ]
    )

    header_cols = (
        ["layer"]
        + [f"chi_s{i+1}" for i in range(max_rank)]
        + [f"target_s{i+1}" for i in range(max_rank)]
    )

    csv_path = os.path.join(
        save_dir,
        f"quimb_schmidt_cut_{cut}_{n_qubits}_qubits.csv",
    )

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=",".join(header_cols),
        comments="",
    )

    print("Saved:", csv_path)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))

    for j in range(max_rank):

        y_chi = chi_matrix[:, j]
        y_target = np.full_like(layers, s_target[j], dtype=float)

        # avoid zero issue on log scale
        y_chi_plot = np.maximum(y_chi, plot_floor)
        y_target_plot = np.maximum(y_target, plot_floor)

        plt.plot(
            layers,
            y_chi_plot,
            marker="o",
            linewidth=1,
            label=f"chi s{j+1}",
        )

        plt.plot(
            layers,
            y_target_plot,
            linestyle="--",
            linewidth=1,
            label=f"target s{j+1}",
        )

    plt.yscale("log")
    plt.xlabel("Number of MPS initializer layers")
    plt.ylabel("Schmidt coefficient value")
    plt.title(
        f"Quimb Schmidt coefficients vs layers\n"
        f"Cut {cut}: {cut} qubits | {n_qubits - cut} qubits"
    )
    plt.grid(True, which="both")

    plt.legend(fontsize=7, ncol=2)

    save_and_show(
        f"quimb_schmidt_coefficients_cut_{cut}_{n_qubits}_qubits.png"
    )


# ==================================================
# Final summary
# ==================================================

print("\nDone.")
print("All plots saved in:", os.path.abspath(save_dir))
print("Summary CSV saved as:", summary_path)