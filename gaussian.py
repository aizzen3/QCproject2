import os
import time

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_mps_initializer.datatypes import QuantumState
from qiskit.quantum_info import state_fidelity


# ============================================================
# FUNCTIONS
# ============================================================

def spatial_grid(L, Nx):
    return np.linspace(-L/2, L/2, Nx, endpoint=False)


def gaussian_thickness_profile(x, d0, h, sigma, x0=0):
    return d0 + h * np.exp(-((x - x0)**2) / (2 * sigma**2))


def phase_signal(d, n, lam):
    k = 2 * np.pi / lam
    return n * k * d


def phi_from_f(f):
    alpha = np.sum(f)
    phi = np.sqrt(f / alpha)
    return phi, alpha


def plot_signal(x, f, n_qubits):
    plt.figure()
    plt.plot(x * 1e6, f)

    plt.xlabel("x (µm)")
    plt.ylabel("f(x) = n k d(x) (rad)")
    plt.title(f"Gaussian Phase Signal ({n_qubits} qubits)")

    plt.grid(True)

    plt.savefig(f"signal_{n_qubits}q.png", dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PARAMETERS
    # ========================================================

    T = 10e-6

    lam = 630e-9
    n_refr = 1.0

    d0 = 100e-9
    h = 200e-9
    sigma = 1e-6
    x0 = 2e-6

    sim = AerSimulator(method="statevector")

    qubit_range = range(5, 10)
    layer_range = range(1, 201)

    threshold = 0.999

    best_layers = []
    best_fidelities = []

    # ========================================================
    # GENERATE SIGNALS
    # ========================================================

    for n_qubits in range(6, 10):

        Nx = 2**n_qubits

        print(f"\nRunning for {n_qubits} qubits (Nx = {Nx})")

        x = spatial_grid(T, Nx)

        d = gaussian_thickness_profile(
            x,
            d0,
            h,
            sigma,
            x0=x0
        )

        f = phase_signal(d, n_refr, lam)

        phi, alpha = phi_from_f(f)

        plot_signal(x, f, n_qubits)

    # ========================================================
    # MPS INITIALIZATION TEST
    # ========================================================

    for n_qubits in qubit_range:

        Nx = 2**n_qubits

        x = spatial_grid(T, Nx)

        d = gaussian_thickness_profile(
            x,
            d0,
            h,
            sigma,
            x0=x0
        )

        f = phase_signal(d, n_refr, lam)

        phi, _ = phi_from_f(f)

        phi = np.asarray(phi, dtype=complex)
        phi = phi / np.linalg.norm(phi)

        print(f"\nRunning for {n_qubits} qubits")

        reached_threshold = False

        for L in layer_range:

            phi_state = QuantumState.from_dense_data(
                data=phi,
                normalize=False
            )

            U_phi = phi_state.generate_mps_initializer_circuit(
                number_of_layers=L
            )

            qc = QuantumCircuit(n_qubits)

            qc.append(U_phi, range(n_qubits))

            qc.save_statevector()

            result = sim.run(
                transpile(qc, sim)
            ).result()

            vec_sim = np.array(
                result.get_statevector(qc),
                dtype=complex
            )

            F = state_fidelity(vec_sim, phi)

            print(f"  Layers = {L}, Fidelity = {F:.6f}")

            if F >= threshold:

                print(
                    f"  ✅ Threshold reached at layer "
                    f"{L}, Fidelity = {F:.6f}"
                )

                best_layers.append(L)
                best_fidelities.append(F)

                reached_threshold = True

                break

        if not reached_threshold:

            print("  ❌ Threshold not reached up to 200 layers")

            best_layers.append(np.nan)
            best_fidelities.append(np.nan)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nSummary:")

    for q, L, F in zip(
        qubit_range,
        best_layers,
        best_fidelities
    ):
        print(f"Qubits = {q}, Best layer = {L}, Fidelity = {F}")

    plt.figure()

    plt.plot(
        list(qubit_range),
        best_layers,
        marker="o"
    )

    plt.xlabel("Number of qubits")
    plt.ylabel("Minimum layers for fidelity ≥ 0.99")

    plt.title("Required MPS layers vs qubit number")

    plt.grid(True)

    plt.savefig("layers_vs_qubits.png", dpi=300)
    plt.close()

    # ========================================================
    # AMPLITUDE COMPARISON
    # ========================================================

    for qi, n_qubits in enumerate(qubit_range):

        chosen_layer = best_layers[qi]

        if np.isnan(chosen_layer):

            print(
                f"{n_qubits} qubits: "
                f"fidelity {threshold} not reached."
            )

            continue

        Nx = 2**n_qubits

        x = spatial_grid(T, Nx)

        d = gaussian_thickness_profile(
            x,
            d0,
            h,
            sigma,
            x0=x0
        )

        f = phase_signal(d, n_refr, lam)

        phi, _ = phi_from_f(f)

        phi = np.asarray(phi, dtype=complex)
        phi = phi / np.linalg.norm(phi)

        phi_state = QuantumState.from_dense_data(
            data=phi,
            normalize=False
        )

        U_phi = phi_state.generate_mps_initializer_circuit(
            number_of_layers=int(chosen_layer)
        )

        qc = QuantumCircuit(n_qubits)

        qc.append(U_phi, range(n_qubits))

        qc.save_statevector()

        result = sim.run(
            transpile(qc, sim)
        ).result()

        vec_sim = np.array(
            result.get_statevector(qc),
            dtype=complex
        )

        global_phase = np.angle(
            np.vdot(phi, vec_sim)
        )

        vec_sim_aligned = vec_sim * np.exp(-1j * global_phase)

        amps_ideal = np.abs(phi)
        amps_sim = np.abs(vec_sim_aligned)

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(Nx),
            amps_ideal,
            'o-',
            label="Ideal amplitude"
        )

        plt.plot(
            range(Nx),
            amps_sim,
            's--',
            label=f"Prepared amplitude, L={int(chosen_layer)}"
        )

        plt.xlabel("Basis index")
        plt.ylabel("Amplitude magnitude")

        plt.title(
            f"{n_qubits} qubits, "
            f"L={int(chosen_layer)}, "
            f"Fidelity ≥ {threshold}"
        )

        plt.legend()
        plt.grid(True)

        plt.savefig(
            f"amplitudes_{n_qubits}q.png",
            dpi=300
        )

        plt.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    main()

    end_time = time.time()

    total_seconds = end_time - start_time

    print("\n==========================")
    print(f"Total runtime: {total_seconds:.2f} seconds")
    print(f"Total runtime: {total_seconds / 60:.2f} minutes")
    print("==========================")