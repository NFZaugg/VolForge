import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import invgauss

from vol_forge.black_scholes_lib.black_scholes.pricing import black_scholes

warnings.filterwarnings("ignore", module="scipy")


def black76(
    F: float,
    K: NDArray[np.float64],
    T: float,
    r: float,
    sigma: NDArray[np.float64],
    is_call: bool = True,
) -> NDArray[np.float64]:
    return black_scholes(F, K, T, r, r, sigma, is_call)


def get_implied_vol_black(
    V: NDArray[np.float64],
    F: float,
    K: NDArray[np.float64],
    tau: float,
    r: float,
    is_call=True,
) -> NDArray[np.float64]:
    D = np.exp(-r * tau)
    if is_call:
        V = V / D / F
    else:
        V = V / D / F + 1 - K / F

    k = np.log(K / F)

    k = np.where(k == 0, 1e-12, k)
    x = np.where(k < 0, (1 - V) / (K / F), (1 - V))
    iv = 2 / np.sqrt(tau) * 1 / np.sqrt(invgauss.ppf(x, mu=2 / np.abs(k)))
    return iv
