import numpy as np
from scipy.stats import invgauss, norm
from numpy.typing import NDArray


def black_scholes_d1d2(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    pc = 1 if is_call else -1
    d1, d2 = black_scholes_d1d2(S, K, T, r, q, sigma)
    return pc * (
        np.exp(-q * T) * S * norm.cdf(pc * d1) - np.exp(-r * T) * K * norm.cdf(pc * d2)
    )


def get_implied_vol_bs(
    V: NDArray[np.float64],
    S: float,
    K: NDArray[np.float64],
    tau: float,
    r: float,
    q: float,
    is_call=True,
) -> NDArray[np.float64]:
    D = np.exp(-r * tau)
    F = np.exp((r - q) * tau) * S
    if is_call:
        V = V / D / F
    else:
        V = V / D / F + 1 - K / F

    k = np.log(K / F)

    k = np.where(k == 0, 1e-12, k)
    x = np.where(k < 0, (1 - V) / (K / F), (1 - V))

    iv = 2 / np.sqrt(tau) * 1 / np.sqrt(invgauss.ppf(x, mu=2 / np.abs(k)))

    return iv
