"""
Unit tests for Black-Scholes greeks.

Strategy:
1. Cross-check every analytic greek against a central finite-difference
   approximation of the price function (`black_scholes`).
2. Cross-check known closed-form identities (put-call parity relationships)
   that must hold regardless of the finite-difference step size.
3. Sanity-check signs/bounds (e.g. gamma > 0, vega > 0, call delta in [0, e^-qT]).
"""

import numpy as np
import pytest

from black_scholes_lib.black_scholes.pricing import black_scholes
from black_scholes_lib.black_scholes.greeks import (
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    bs_vanna,
    bs_volga,
)


# ---------------------------------------------------------------------------
# Fixtures / shared test parameters
# ---------------------------------------------------------------------------

S0 = 100.0
T0 = 0.75
R0 = 0.03
Q0 = 0.015

STRIKES = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
SIGMAS = np.array([0.35, 0.28, 0.22, 0.25, 0.30])

FD_EPS = 1e-4
FD_EPS_T = 1e-5
FD_EPS_R = 1e-5
ATOL = 1e-3
RTOL = 1e-3


def _fd_derivative(f, x0, eps, relative=True):
    h = eps * x0 if relative else eps
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------


class TestDelta:
    @pytest.mark.parametrize("is_call", [True, False])
    def test_delta_matches_finite_difference(self, is_call):
        # when
        analytic = bs_delta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call)
        fd = _fd_derivative(
            lambda S: black_scholes(S, STRIKES, T0, R0, Q0, SIGMAS, is_call), S0, FD_EPS
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_delta_put_call_parity(self):
        # given
        # when
        call_delta = bs_delta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=True)
        put_delta = bs_delta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=False)

        # then
        np.testing.assert_allclose(
            call_delta - put_delta,
            np.full_like(call_delta, np.exp(-Q0 * T0)),
            rtol=1e-10,
        )

    def test_call_delta_bounds(self):
        # given
        # when
        delta = bs_delta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=True)

        # then
        assert np.all(delta >= 0.0)
        assert np.all(delta <= np.exp(-Q0 * T0) + 1e-12)

    def test_put_delta_bounds(self):
        # given
        # when
        delta = bs_delta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=False)

        # then
        assert np.all(delta <= 0.0)
        assert np.all(delta >= -np.exp(-Q0 * T0) - 1e-12)


class TestGamma:
    def test_gamma_matches_finite_difference_of_delta(self):

        # when
        analytic = bs_gamma(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd = _fd_derivative(
            lambda S: bs_delta(S, STRIKES, T0, R0, Q0, SIGMAS, is_call=True), S0, FD_EPS
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_gamma_same_for_call_and_put(self):
        # when
        fd_call = _fd_derivative(
            lambda S: bs_delta(S, STRIKES, T0, R0, Q0, SIGMAS, is_call=True), S0, FD_EPS
        )
        fd_put = _fd_derivative(
            lambda S: bs_delta(S, STRIKES, T0, R0, Q0, SIGMAS, is_call=False),
            S0,
            FD_EPS,
        )

        # then
        np.testing.assert_allclose(fd_call, fd_put, rtol=RTOL, atol=ATOL)

    def test_gamma_is_positive(self):
        # given
        # when
        gamma = bs_gamma(S0, STRIKES, T0, R0, Q0, SIGMAS)

        # then
        assert np.all(gamma > 0.0)


class TestVega:
    @pytest.mark.parametrize("is_call", [True, False])
    def test_vega_matches_finite_difference(self, is_call):

        # when
        analytic = bs_vega(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd = _fd_derivative(
            lambda sig: black_scholes(S0, STRIKES, T0, R0, Q0, sig, is_call),
            SIGMAS,
            FD_EPS,
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_vega_is_positive(self):
        # given
        # when
        vega = bs_vega(S0, STRIKES, T0, R0, Q0, SIGMAS)

        # then
        assert np.all(vega > 0.0)

    def test_vega_same_for_call_and_put(self):

        # when
        fd_call = _fd_derivative(
            lambda sig: black_scholes(S0, STRIKES, T0, R0, Q0, sig, is_call=True),
            SIGMAS,
            FD_EPS,
        )
        fd_put = _fd_derivative(
            lambda sig: black_scholes(S0, STRIKES, T0, R0, Q0, sig, is_call=False),
            SIGMAS,
            FD_EPS,
        )

        # then
        np.testing.assert_allclose(fd_call, fd_put, rtol=RTOL, atol=ATOL)


class TestTheta:
    @pytest.mark.parametrize("is_call", [True, False])
    def test_theta_matches_finite_difference(self, is_call):

        # when
        analytic = bs_theta(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call)
        fd_theta = -_fd_derivative(
            lambda T: black_scholes(S0, STRIKES, T, R0, Q0, SIGMAS, is_call),
            T0,
            FD_EPS_T,
            relative=False,
        )

        # then
        np.testing.assert_allclose(analytic, fd_theta, rtol=RTOL, atol=ATOL)


class TestRho:
    @pytest.mark.parametrize("is_call", [True, False])
    def test_rho_matches_finite_difference(self, is_call):
        # when
        analytic = bs_rho(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call)
        fd = _fd_derivative(
            lambda r: black_scholes(S0, STRIKES, T0, r, Q0, SIGMAS, is_call),
            R0,
            FD_EPS_R,
            relative=False,
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_rho_put_call_parity(self):
        # when
        call_rho = bs_rho(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=True)
        put_rho = bs_rho(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=False)
        expected_diff = STRIKES * T0 * np.exp(-R0 * T0)

        # then
        np.testing.assert_allclose(
            call_rho - put_rho, expected_diff, rtol=1e-6, atol=1e-8
        )

    def test_call_rho_is_positive_put_rho_is_negative(self):
        # when
        call_rho = bs_rho(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=True)
        put_rho = bs_rho(S0, STRIKES, T0, R0, Q0, SIGMAS, is_call=False)

        # then
        assert np.all(call_rho > 0.0)
        assert np.all(put_rho < 0.0)


class TestVanna:
    def test_vanna_matches_finite_difference_of_vega_wrt_S(self):

        # when
        analytic = bs_vanna(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd = _fd_derivative(
            lambda S: bs_vega(S, STRIKES, T0, R0, Q0, SIGMAS), S0, FD_EPS
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_vanna_matches_finite_difference_of_delta_wrt_sigma(self):

        # when
        analytic = bs_vanna(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd = _fd_derivative(
            lambda sig: bs_delta(S0, STRIKES, T0, R0, Q0, sig, is_call=True),
            SIGMAS,
            FD_EPS,
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_vanna_same_for_call_and_put(self):

        # when
        analytic = bs_vanna(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd_put = _fd_derivative(
            lambda sig: bs_delta(S0, STRIKES, T0, R0, Q0, sig, is_call=False),
            SIGMAS,
            FD_EPS,
        )

        # then
        np.testing.assert_allclose(analytic, fd_put, rtol=RTOL, atol=ATOL)


class TestVolga:
    def test_volga_matches_finite_difference_of_vega_wrt_sigma(self):

        # when
        analytic = bs_volga(S0, STRIKES, T0, R0, Q0, SIGMAS)
        fd = _fd_derivative(
            lambda sig: bs_vega(S0, STRIKES, T0, R0, Q0, sig), SIGMAS, FD_EPS
        )

        # then
        np.testing.assert_allclose(analytic, fd, rtol=RTOL, atol=ATOL)

    def test_volga_zero_at_d1_or_d2_zero(self):
        # given
        S = 100.0
        T = 1.0
        r = q = 0.0
        K = np.array([105.0])
        sigma = np.sqrt(-2 * np.log(S / K) / T)

        # when
        volga = bs_volga(S, K, T, r, q, sigma)

        # then
        np.testing.assert_allclose(volga, 0.0, atol=1e-8)


class TestOutputShape:
    @pytest.mark.parametrize(
        "greek_fn,needs_is_call",
        [
            (bs_delta, True),
            (bs_gamma, False),
            (bs_vega, False),
            (bs_theta, True),
            (bs_rho, True),
            (bs_vanna, False),
            (bs_volga, False),
        ],
    )
    def test_output_shape_matches_input(self, greek_fn, needs_is_call):
        # given
        kwargs = dict(S=S0, K=STRIKES, T=T0, r=R0, q=Q0, sigma=SIGMAS)
        if needs_is_call:
            kwargs["is_call"] = True

        # when
        result = greek_fn(**kwargs)

        # then
        assert result.shape == STRIKES.shape
        assert np.all(np.isfinite(result))
