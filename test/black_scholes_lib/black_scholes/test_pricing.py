import numpy as np
from vol_forge.black_scholes_lib.black.pricing import (
    black_scholes,
)
from vol_forge.black_scholes_lib.black_scholes.pricing import get_implied_vol_bs


class TestBlackScholesPricing:
    def test_black_scholes_call_put_parity(self) -> None:
        # Given: matching call and put inputs
        S = 100.0
        q = 0.015
        r = 0.03
        T = 0.5
        K = np.array([90.0, 100.0, 110.0])
        sigma = np.array([0.2, 0.2, 0.2])

        # When: pricing calls and puts at the same strikes
        call = black_scholes(S, K, T, r, q, sigma, is_call=True)
        put = black_scholes(S, K, T, r, q, sigma, is_call=False)

        # Then: put-call parity should hold: C - P = S*e^(-qT) - K*e^(-rT)
        expected_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
        np.testing.assert_allclose(call - put, expected_diff, rtol=1e-10, atol=1e-12)

    def test_get_implied_vol_bs_recovers_input_vol(self) -> None:
        # Given: known true vols used to generate synthetic option prices via black_scholes
        S = 1.1
        q = 0.01
        r = 0.02
        tau = 1.0
        K = np.array([0.9, 1.0, 1.2])
        true_sigma = np.array([0.3, 0.25, 0.35])
        V = black_scholes(S, K, tau, r, q, true_sigma, is_call=True)

        # When: inverting those prices back to implied vols
        recovered_sigma = get_implied_vol_bs(V, S, K, tau, r, q, is_call=True)

        # Then: the recovered vols should match the true vols within a reasonable tolerance
        np.testing.assert_allclose(recovered_sigma, true_sigma, rtol=1e-2, atol=1e-3)

    def test_get_implied_vol_bs_uses_tau_not_global_T(self) -> None:
        # Given: a tau that differs from any module-level T, to catch regressions
        # of the T-vs-tau bug where the function used a global `T` instead of its `tau` argument
        S = 1.1
        q = 0.01
        r = 0.02
        tau = 0.35
        K = np.array([1.0])
        true_sigma = np.array([0.2])
        V = black_scholes(S, K, tau, r, q, true_sigma, is_call=False)

        # When: computing implied vol at this specific tau
        recovered_sigma = get_implied_vol_bs(V, S, K, tau, r, q, is_call=False)

        # Then: it should still recover the true vol (fails if `tau` argument is ignored)
        np.testing.assert_allclose(recovered_sigma, true_sigma, rtol=1e-2, atol=1e-3)
