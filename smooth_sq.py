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
# Functions
# ============================================================

def spatial_grid(L, Nx):
    return np.linspace(-L / 2, L / 2, Nx, endpoint=True)


def thickness_profile_sharp(x, Lambda, duty, d0, h):
    phase = (x % Lambda) / Lambda
    return d0 + h * (phase < duty).astype(float)


def thickness_profile_smooth_periodic(x, Lambda, duty, d0, h, k=40):
    phase = (x % Lambda) / Lambda

    center = duty / 2
    u = phase - center
    u = (u + 0.5) % 1.0 - 0.5

    halfwidth = duty / 2
    pulse = 0.5 * (1 - np.tanh(k * (np.abs(u) - halfwidth)))

    return d0 + h * pulse


def phase_signal(d, n_refr, lam):
    k0 = 2 * np.pi / lam
    return n_refr * k0 * d


def phi_from_f(f):
    alpha = np.sum(f)
    phi = np.sqrt(f / alpha)
    return phi, alpha


def exp_func(x, A, B):
    return A * np.exp(B * x)


# Main


def main():

    os.makedirs("results_smooth", exist_ok=True)

    sim = AerSimulator(method="statevector")

    # ----------------------------
    # Parameters
    # ----------------------------

    T = 16e-6

    lam = 630e-9
    n_refr = 1.0

    Lambda = 2e-6
    duty = 0.50
    d0 = 0
    h = 200e-9
    smoothness_k = 20

    Nx_plot = 128

    qubit_range = range(5, 10)      # 5 to 11 qubits
    layer_range = range(1, 301)     # 1 to 300 layers
    threshold = 0.999

    best_layers = []
    best_fidelities = []

    
    # Plot sharp vs smooth profile
    

    x = spatial_grid(T, Nx_plot)

    d_sharp = thickness_profile_sharp(
        x=x,
        Lambda=Lambda,
        duty=duty,
        d0=d0,
        h=h
    )

    d_smooth = thickness_profile_smooth_periodic(
        x=x,
        Lambda=Lambda,
        duty=duty,
        d0=d0,
        h=h,
        k=smoothness_k
    )

    f_sharp = phase_signal(d_sharp, n_refr, lam)
    f_smooth = phase_signal(d_smooth, n_refr, lam)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x * 1e6, d_sharp * 1e9, label="Sharp thickness", linewidth=1.5)
    plt.plot(x * 1e6, d_smooth * 1e9, label="Smooth thickness", linewidth=2)
    plt.xlabel("x (µm)")
    plt.ylabel("d(x) (nm)")
    plt.title("Thickness profile")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x * 1e6, f_sharp, label="Sharp phase", linewidth=1.5)
    plt.plot(x * 1e6, f_smooth, label="Smooth phase", linewidth=2)
    plt.xlabel("x (µm)")
    plt.ylabel("f(x) = n k d(x) (rad)")
    plt.title("Phase signal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results_smooth/sharp_vs_smooth_profile.png", dpi=300, bbox_inches="tight")
    plt.close()


    
    # Scan layers with early stopping
    

    for n_qubits in qubit_range:

        Nx = 2 ** n_qubits
        x = spatial_grid(T, Nx)

        d = thickness_profile_smooth_periodic(
            x=x,
            Lambda=Lambda,
            duty=duty,
            d0=d0,
            h=h,
            k=smoothness_k
        )

        f = phase_signal(d, n_refr, lam)
        phi, _ = phi_from_f(f)

        phi = np.asarray(phi, dtype=complex)
        phi = phi / np.linalg.norm(phi)

        phi_state = QuantumState.from_dense_data(data=phi, normalize=False)

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
            vec_sim = np.array(result.get_statevector(tqc), dtype=complex)

            F = state_fidelity(vec_sim, phi)

            print(f"  Layers = {L}, Fidelity = {F:.6f}")

            if F >= threshold:
                print(f"  Threshold reached at layer {L}, Fidelity = {F:.6f}")
                best_layers.append(L)
                best_fidelities.append(F)
                reached_threshold = True
                break

        if not reached_threshold:
            print("  Threshold not reached up to 300 layers")
            best_layers.append(np.nan)
            best_fidelities.append(np.nan)

    
    # Save numerical results
    

    results = np.column_stack([
        np.array(list(qubit_range), dtype=float),
        np.array(best_layers, dtype=float),
        np.array(best_fidelities, dtype=float)
    ])

    np.savetxt(
        "results_smooth/mps_smooth_results.csv",
        results,
        delimiter=",",
        header="qubits,best_layer,fidelity",
        comments=""
    )

    print("\nSaved results to results_smooth/mps_smooth_results.csv")

    print("\nSummary")
    for q, L_best, F_best in zip(qubit_range, best_layers, best_fidelities):
        print(f"Qubits = {q}, Layer = {L_best}, Fidelity = {F_best}")

    
    # Fit scaling
    

    qubits = np.array(list(qubit_range), dtype=float)
    layers = np.array(best_layers, dtype=float)
    fids = np.array(best_fidelities, dtype=float)

    mask = np.isfinite(layers)

    x_fit = qubits[mask]
    y_fit = layers[mask]
    fid_fit = fids[mask]

    print("\nUsed for fitting:")
    for q, L, F in zip(x_fit, y_fit, fid_fit):
        print(f"Qubits = {int(q)}, Layers = {L:.0f}, Fidelity = {F:.6f}")

    if len(x_fit) >= 3:

        linear_coeff = np.polyfit(x_fit, y_fit, 1)
        linear_model = np.poly1d(linear_coeff)
        y_linear = linear_model(x_fit)

        quad_coeff = np.polyfit(x_fit, y_fit, 2)
        quad_model = np.poly1d(quad_coeff)
        y_quad = quad_model(x_fit)

        params, _ = curve_fit(exp_func, x_fit, y_fit, maxfev=10000)
        y_exp = exp_func(x_fit, *params)

        r2_linear = r2_score(y_fit, y_linear)
        r2_quad = r2_score(y_fit, y_quad)
        r2_exp = r2_score(y_fit, y_exp)

        print("\nFit quality:")
        print("Linear R² =", r2_linear)
        print("Quadratic R² =", r2_quad)
        print("Exponential R² =", r2_exp)

        plt.figure(figsize=(8, 6))

        plt.scatter(x_fit, y_fit, s=80, label="Data")
        plt.plot(x_fit, y_linear, label=f"Linear R²={r2_linear:.3f}")
        plt.plot(x_fit, y_quad, label=f"Quadratic R²={r2_quad:.3f}")
        plt.plot(x_fit, y_exp, label=f"Exponential R²={r2_exp:.3f}")

        for q, L, F in zip(x_fit, y_fit, fid_fit):
            plt.text(q, L + 1, f"{F:.3f}", ha="center", fontsize=9)

        plt.xlabel("Number of qubits")
        plt.ylabel(f"Number of MPS layers needed for fidelity ≥ {threshold}")
        plt.title("Scaling of Required MPS Layers")
        plt.xticks(x_fit.astype(int))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results_smooth/scaling_fit_smooth.png", dpi=300, bbox_inches="tight")
        plt.close()


    else:
        print("\nNot enough valid data points for fitting.")

    
    # Amplitude comparison plots
    

    for qi, n_qubits in enumerate(qubit_range):

        chosen_layer = best_layers[qi]

        if np.isnan(chosen_layer):
            print(f"{n_qubits} qubits: threshold not reached.")
            continue

        Nx = 2 ** n_qubits
        x = spatial_grid(T, Nx)

        d = thickness_profile_smooth_periodic(
            x=x,
            Lambda=Lambda,
            duty=duty,
            d0=d0,
            h=h,
            k=smoothness_k
        )

        f = phase_signal(d, n_refr, lam)
        phi, _ = phi_from_f(f)

        phi = np.asarray(phi, dtype=complex)
        phi = phi / np.linalg.norm(phi)

        phi_state = QuantumState.from_dense_data(data=phi, normalize=False)

        U_phi = phi_state.generate_mps_initializer_circuit(
            number_of_layers=int(chosen_layer)
        )

        qc = QuantumCircuit(n_qubits)
        qc.append(U_phi, range(n_qubits))
        qc.save_statevector()

        tqc = transpile(qc, sim)
        result = sim.run(tqc).result()
        vec_sim = np.array(result.get_statevector(tqc), dtype=complex)

        global_phase = np.angle(np.vdot(phi, vec_sim))
        vec_sim_aligned = vec_sim * np.exp(-1j * global_phase)

        amps_ideal = np.abs(phi)
        amps_sim = np.abs(vec_sim_aligned)

        plt.figure(figsize=(8, 5))

        plt.plot(range(Nx), amps_ideal, "o-", label="Ideal amplitude")
        plt.plot(
            range(Nx),
            amps_sim,
            "s--",
            label=f"Prepared amplitude, L={int(chosen_layer)}"
        )

        plt.xlabel("Basis index")
        plt.ylabel("Amplitude magnitude")
        plt.title(
            f"{n_qubits} qubits | Layer={int(chosen_layer)} | Fidelity ≥ {threshold}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"results_smooth/amplitude_{n_qubits}q_smooth.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()




# Run script and measure time


if __name__ == "__main__":

    start_time = time.time()

    main()

    end_time = time.time()
    total_seconds = end_time - start_time

    print("\n==========================")
    print(f"Total runtime: {total_seconds:.2f} seconds")
    print(f"Total runtime: {total_seconds / 60:.2f} minutes")
    print("==========================")