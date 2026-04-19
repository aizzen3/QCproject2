import numpy as np

def gaussian_beam(x: np.ndarray, w0: float) -> np.ndarray:
    return np.exp(-(x**2) / (w0**2)).astype(np.complex128)


def propagate_free_space(
    psi_x: np.ndarray,
    dx: float,
    wavelength: float,
    z: float
) -> np.ndarray:
    fx = np.fft.fftfreq(len(psi_x), d=dx)
    kx = 2 * np.pi * fx
    k0 = 2 * np.pi / wavelength

    H = np.exp(-1j * (kx**2) * z / (2 * k0))

    psi_k = np.fft.fft(psi_x, norm="ortho")
    psi_out = np.fft.ifft(psi_k * H, norm="ortho")
    return psi_out


def simulate_gaussian(n_qubits, Lx, w0, wavelength, z):
    N = 2**n_qubits
    x = np.linspace(-Lx / 2, Lx / 2, N, endpoint=False)
    dx = x[1] - x[0]

    psi0 = gaussian_beam(x, w0)
    psi0 = psi0 / np.linalg.norm(psi0)

    psi_z = propagate_free_space(psi0, dx, wavelength, z)
    psi_z = psi_z / np.linalg.norm(psi_z)

    return x, dx, psi0, psi_z