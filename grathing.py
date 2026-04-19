import numpy as np


def spatial_grid(L, Nx):
    return np.linspace(-L/2, L/2, Nx, endpoint=False)


def thickness_profile(x, Lambda, duty, d0, h):
    phase_in_period = (x % Lambda) / Lambda
    d = d0 + h * (phase_in_period < duty).astype(float)
    return d


def phase_signal(d, n, lam):
    k = 2 * np.pi / lam
    return n * k * d


def phi_from_f(f):
    alpha = np.sum(f)
    phi = np.sqrt(f / alpha)
    return phi, alpha


# 🔥 Main function (THIS is what you will import)
def generate_phi(L, Nx, lam, n, Lambda, duty, d0, h):
    x = spatial_grid(L, Nx)
    d = thickness_profile(x, Lambda, duty, d0, h)
    f = phase_signal(d, n, lam)
    phi, alpha = phi_from_f(f)

    return x, f, phi, alpha