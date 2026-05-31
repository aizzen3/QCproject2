import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_mps_initializer.datatypes import QuantumState
from qiskit.quantum_info import state_fidelity


# ============================================================
# FUNCTIONS
# ============================================================

def spatial_grid(L, Nx):

    return np.linspace(-L / 2, L / 2, Nx, endpoint=False)


def gaussian_thickness_profile(
    x,
    d0,
    h,
    sigma,
    x0=0
):

    return d0 + h * np.exp(
        -((x - x0) ** 2) / (2 * sigma ** 2)
    )


def phase_signal(d, n_refr, lam):

    k0 = 2 * np.pi / lam

    return n_refr * k0 * d


def phi_from_f(f):

    alpha = np.sum(f)

    phi = np.sqrt(f / alpha)

    return phi, alpha


def exp_func(x, A, B):

    return A * np.exp(B * x)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CREATE RESULTS DIRECTORY
    # ========================================================

    RESULTS_DIR = "results_non_shifted_gaussian"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sim = AerSimulator(method="statevector")

    # ========================================================
    # PARAMETERS
    # ========================================================

    T = 10e-6

    lam = 630e-9
    n_refr = 1.0

    d0 = 0
    h = 200e-9

    sigma = 1e-6
    x0 = 0

    Nx_plot = 256

    qubit_range = range(5, 12)
    layer_range = range(1, 301)

    threshold = 0.999

    best_layers = []
    best_fidelities = []

    # ========================================================
    # PLOT GAUSSIAN PROFILE
    # ========================================================

    x = spatial_grid(T, Nx_plot)

    d = gaussian_thickness_profile(
        x=x,
        d0=d0,
        h=h,
        sigma=sigma,
        x0=x0
    )

    f = phase_signal(d, n_refr, lam)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.plot(
        x * 1e6,
        d * 1e9,
        linewidth=2
    )

    plt.xlabel("x (µm)")
    plt.ylabel("d(x) (nm)")

    plt.title("Gaussian Thickness Profile")

    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(
        x * 1e6,
        f,
        linewidth=2
    )

    plt.xlabel("x (µm)")
    plt.ylabel("f(x) = n k d(x) (rad)")

    plt.title("Gaussian Phase Signal")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results_non_shifted_gaussian/gaussian_profile.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SCAN LAYERS WITH EARLY STOPPING
    # ========================================================

    for n_qubits in qubit_range:

        Nx = 2 ** n_qubits

        x = spatial_grid(T, Nx)

        d = gaussian_thickness_profile(
            x=x,
            d0=d0,
            h=h,
            sigma=sigma,
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

        print(f"\nRunning for {n_qubits} qubits")

        reached_threshold = False

        for L in layer_range:

            U_phi = phi_state.generate_mps_initializer_circuit(
                number_of_layers=L
            )

            qc = QuantumCircuit(n_qubits)

            qc.append(U_phi, range(n_qubits))

            qc.save_statevector()

            tqc = transpile(qc, sim)

            result = sim.run(tqc).result()

            vec_sim = np.array(
                result.get_statevector(tqc),
                dtype=complex
            )

            F = state_fidelity(vec_sim, phi)

            print(f"  Layers = {L}, Fidelity = {F:.6f}")

            if F >= threshold:

                print(
                    f"  Threshold reached at layer "
                    f"{L}, Fidelity = {F:.6f}"
                )

                best_layers.append(L)

                best_fidelities.append(F)

                reached_threshold = True

                break

        if not reached_threshold:

            print(
                "  Threshold not reached "
                "up to 400 layers"
            )

            best_layers.append(np.nan)

            best_fidelities.append(np.nan)

    # ========================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================

    results = np.column_stack([
        np.array(list(qubit_range), dtype=float),
        np.array(best_layers, dtype=float),
        np.array(best_fidelities, dtype=float)
    ])

    np.savetxt(
        "results_non_shifted_gaussian/mps_gaussian_results.csv",
        results,
        delimiter=",",
        header="qubits,best_layer,fidelity",
        comments=""
    )

    print(
        "\nSaved results to "
        "results_non_shifted_gaussian/mps_gaussian_results.csv"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\nSummary")

    for q, L_best, F_best in zip(
        qubit_range,
        best_layers,
        best_fidelities
    ):

        print(
            f"Qubits = {q}, "
            f"Layer = {L_best}, "
            f"Fidelity = {F_best}"
        )

    # ========================================================
    # SAVE SUMMARY TXT
    # ========================================================

    with open(
        "results_non_shifted_gaussian/summary.txt",
        "w"
    ) as file:

        file.write(
            "Gaussian MPS Initialization Results\n"
        )

        file.write(
            "===================================\n\n"
        )

        file.write(
            f"Threshold fidelity = {threshold}\n\n"
        )

        for q, L_best, F_best in zip(
            qubit_range,
            best_layers,
            best_fidelities
        ):

            file.write(
                f"Qubits = {q}, "
                f"Layer = {L_best}, "
                f"Fidelity = {F_best}\n"
            )

    # ========================================================
    # FIT SCALING
    # ========================================================

    qubits = np.array(
        list(qubit_range),
        dtype=float
    )

    layers = np.array(
        best_layers,
        dtype=float
    )

    fids = np.array(
        best_fidelities,
        dtype=float
    )

    mask = np.isfinite(layers)

    x_fit = qubits[mask]

    y_fit = layers[mask]

    fid_fit = fids[mask]

    print("\nUsed for fitting:")

    for q, L, F in zip(
        x_fit,
        y_fit,
        fid_fit
    ):

        print(
            f"Qubits = {int(q)}, "
            f"Layers = {L:.0f}, "
            f"Fidelity = {F:.6f}"
        )

    if len(x_fit) >= 3:

        # ====================================================
        # LINEAR FIT
        # ====================================================

        linear_coeff = np.polyfit(
            x_fit,
            y_fit,
            1
        )

        linear_model = np.poly1d(linear_coeff)

        y_linear = linear_model(x_fit)

        # ====================================================
        # QUADRATIC FIT
        # ====================================================

        quad_coeff = np.polyfit(
            x_fit,
            y_fit,
            2
        )

        quad_model = np.poly1d(quad_coeff)

        y_quad = quad_model(x_fit)

        # ====================================================
        # EXPONENTIAL FIT
        # ====================================================

        params, _ = curve_fit(
            exp_func,
            x_fit,
            y_fit,
            maxfev=10000
        )

        y_exp = exp_func(
            x_fit,
            *params
        )

        # ====================================================
        # R² SCORES
        # ====================================================

        r2_linear = r2_score(
            y_fit,
            y_linear
        )

        r2_quad = r2_score(
            y_fit,
            y_quad
        )

        r2_exp = r2_score(
            y_fit,
            y_exp
        )

        print("\nFit quality:")

        print("Linear R² =", r2_linear)

        print("Quadratic R² =", r2_quad)

        print("Exponential R² =", r2_exp)

        # ====================================================
        # PLOT FITS
        # ====================================================

        plt.figure(figsize=(8, 6))

        plt.scatter(
            x_fit,
            y_fit,
            s=80,
            label="Data"
        )

        plt.plot(
            x_fit,
            y_linear,
            label=f"Linear R²={r2_linear:.3f}"
        )

        plt.plot(
            x_fit,
            y_quad,
            label=f"Quadratic R²={r2_quad:.3f}"
        )

        plt.plot(
            x_fit,
            y_exp,
            label=f"Exponential R²={r2_exp:.3f}"
        )

        for q, L, F in zip(
            x_fit,
            y_fit,
            fid_fit
        ):

            plt.text(
                q,
                L + 1,
                f"{F:.3f}",
                ha="center",
                fontsize=9
            )

        plt.xlabel("Number of qubits")

        plt.ylabel(
            f"Number of MPS layers needed "
            f"for fidelity ≥ {threshold}"
        )

        plt.title(
            "Scaling of Required MPS Layers"
        )

        plt.xticks(x_fit.astype(int))

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "results_non_shifted_gaussian/scaling_fit_gaussian.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    else:

        print(
            "\nNot enough valid data "
            "points for fitting."
        )

    # ========================================================
    # AMPLITUDE COMPARISON PLOTS
    # ========================================================

    for qi, n_qubits in enumerate(qubit_range):

        chosen_layer = best_layers[qi]

        if np.isnan(chosen_layer):

            print(
                f"{n_qubits} qubits: "
                f"threshold not reached."
            )

            continue

        Nx = 2 ** n_qubits

        x = spatial_grid(T, Nx)

        d = gaussian_thickness_profile(
            x=x,
            d0=d0,
            h=h,
            sigma=sigma,
            x0=x0
        )

        f = phase_signal(d, n_refr, lam)

        phi, _ = phi_from_f(f)

        phi = np.asarray(
            phi,
            dtype=complex
        )

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

        tqc = transpile(qc, sim)

        result = sim.run(tqc).result()

        vec_sim = np.array(
            result.get_statevector(tqc),
            dtype=complex
        )

        global_phase = np.angle(
            np.vdot(phi, vec_sim)
        )

        vec_sim_aligned = vec_sim * np.exp(
            -1j * global_phase
        )

        amps_ideal = np.abs(phi)

        amps_sim = np.abs(vec_sim_aligned)

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(Nx),
            amps_ideal,
            "o-",
            label="Ideal amplitude"
        )

        plt.plot(
            range(Nx),
            amps_sim,
            "s--",
            label=(
                f"Prepared amplitude, "
                f"L={int(chosen_layer)}"
            )
        )

        plt.xlabel("Basis index")

        plt.ylabel("Amplitude magnitude")

        plt.title(
            f"{n_qubits} qubits | "
            f"Layer={int(chosen_layer)} | "
            f"Fidelity ≥ {threshold}"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            f"results_non_shifted_gaussian/"
            f"amplitude_{n_qubits}q_gaussian.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# RUN SCRIPT + MEASURE TIME
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    main()

    end_time = time.time()

    total_seconds = end_time - start_time

    with open(
        "results_non_shifted_gaussian/runtime.txt",
        "w"
    ) as file:

        file.write(
            f"Total runtime: "
            f"{total_seconds:.2f} seconds\n"
        )

        file.write(
            f"Total runtime: "
            f"{total_seconds / 60:.2f} minutes\n"
        )

    print("\n==========================")

    print(
        f"Total runtime: "
        f"{total_seconds:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_seconds / 60:.2f} minutes"
    )

    print("==========================")