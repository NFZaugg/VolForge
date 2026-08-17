import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from vol_forge.black_scholes_lib.black_scholes.pricing import black_scholes_d1d2


def bs_delta(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    pc = 1 if is_call else -1
    d1, _ = black_scholes_d1d2(S, K, T, r, q, sigma)
    return pc * np.exp(-q * T) * norm.cdf(pc * d1)


def bs_gamma(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    d1, _ = black_scholes_d1d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    d1, _ = black_scholes_d1d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def bs_theta(
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

    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -pc * r * K * np.exp(-r * T) * norm.cdf(pc * d2)
    term3 = pc * q * S * np.exp(-q * T) * norm.cdf(pc * d1)
    return term1 + term2 + term3


def bs_rho(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    pc = 1 if is_call else -1
    _, d2 = black_scholes_d1d2(S, K, T, r, q, sigma)

    return pc * K * T * np.exp(-r * T) * norm.cdf(pc * d2)


def bs_vanna(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    d1, d2 = black_scholes_d1d2(S, K, T, r, q, sigma)

    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def bs_volga(
    S: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    q: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    d1, d2 = black_scholes_d1d2(S, K, T, r, q, sigma)

    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    return vega * d1 * d2 / sigma
