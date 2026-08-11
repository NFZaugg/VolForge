import numpy as np
from numpy.typing import NDArray

from black_scholes_lib.black_scholes.greeks import (
    bs_delta,
    bs_gamma,
    bs_rho,
    bs_theta,
    bs_vanna,
    bs_vega,
    bs_volga,
)


def black_delta(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    return bs_delta(F, K, T, r, r, sigma, is_call)


def black_gamma(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    return bs_gamma(F, K, T, r, r, sigma)


def black_vega(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    return bs_vega(F, K, T, r, r, sigma)


def black_theta(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    return bs_theta(F, K, T, r, r, sigma, is_call)


def black_rho(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    return bs_rho(F, K, T, r, r, sigma, is_call)


def black_vanna(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    return bs_vanna(F, K, T, r, r, sigma)


def black_volga(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    return bs_volga(F, K, T, r, r, sigma)
