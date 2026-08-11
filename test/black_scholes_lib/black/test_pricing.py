import numpy as np

from black_scholes_lib.black.pricing import black76, get_implied_vol_black
from black_scholes_lib.black_scholes.pricing import black_scholes


class TestBlackPricing:
    def test_black76_matches_black_scholes_with_consistent_forward(self) -> None:
        # Given: a spot, rate, dividend yield, and the forward built consistently as F = S * exp((r-q)*T)
        S = 1.1
        q = 0.01
        r = 0.02
        T = 1.0
        K = np.array([0.9, 1.0, 1.2])
        sigma = np.array([0.3, 0.25, 0.35])
        F = S * np.exp((r - q) * T)

        # When: pricing the same options via black_scholes(spot) and black76(forward)
        bs_price = black_scholes(S, K, T, r, q, sigma, is_call=True)
        b76_price = black76(F, K, T, r, sigma, is_call=True)

        # Then: the two prices should agree to floating point precision
        np.testing.assert_allclose(bs_price, b76_price, rtol=1e-10, atol=1e-12)

    def test_black76_diverges_from_black_scholes_with_inconsistent_forward(
        self,
    ) -> None:
        # Given: a forward that is NOT built as S * exp((r-q)*T) (the classic bug)
        S = 1.1
        q = 0.01
        r = 0.02
        T = 1.0
        K = np.array([0.9, 1.0, 1.2])
        sigma = np.array([0.3, 0.25, 0.35])
        wrong_F = np.exp((r - q) * T)  # missing the S multiplier

        # When: pricing via black_scholes(spot) and black76(wrong forward)
        bs_price = black_scholes(S, K, T, r, q, sigma, is_call=True)
        b76_price = black76(wrong_F, K, T, r, sigma, is_call=True)

        # Then: prices should NOT match, confirming the forward-consistency requirement
        assert not np.allclose(bs_price, b76_price, rtol=1e-6)

    def test_black76_call_put_parity(self) -> None:
        # Given: a forward price and matching call/put inputs
        F = 105.0
        r = 0.02
        T = 1.0
        K = np.array([95.0, 105.0, 115.0])
        sigma = np.array([0.25, 0.25, 0.25])

        # When: pricing calls and puts at the same strikes
        call = black76(F, K, T, r, sigma, is_call=True)
        put = black76(F, K, T, r, sigma, is_call=False)

        # Then: put-call parity for Black-76 should hold: C - P = e^(-rT) * (F - K)
        expected_diff = np.exp(-r * T) * (F - K)
        np.testing.assert_allclose(call - put, expected_diff, rtol=1e-10, atol=1e-12)

    def test_black76_atm_price_matches_known_formula(self) -> None:
        # Given: an at-the-money option (F == K), where the Black-76 formula simplifies
        F = 100.0
        K = np.array([100.0])
        r = 0.01
        T = 1.0
        sigma = np.array([0.2])

        # When: pricing the ATM call
        price = black76(F, K, T, r, sigma, is_call=True)

        # Then: ATM Black-76 call price equals F * exp(-rT) * (2*N(0.5*sigma*sqrt(T)) - 1)
        half_vol_term = 0.5 * sigma[0] * np.sqrt(T)
        from scipy.stats import norm

        expected = F * np.exp(-r * T) * (2 * norm.cdf(half_vol_term) - 1)
        np.testing.assert_allclose(price, expected, rtol=1e-10, atol=1e-12)

    def test_imply_vol_black_round_trip(self) -> None:
        # Given: a fwd and an ir rate
        F = 1.1
        r = 0.02
        tau = 0.35
        K = np.array([1.0])
        true_sigma = np.array([0.2])
        V = black76(F, K, tau, r, true_sigma, is_call=False)

        # When: computing implied vol at this specific tau
        recovered_sigma = get_implied_vol_black(V, F, K, tau, r, is_call=False)

        # Then: it should still recover the true vol (fails if `tau` argument is ignored)
        np.testing.assert_allclose(recovered_sigma, true_sigma, rtol=1e-2, atol=1e-3)
