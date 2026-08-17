from collections.abc import Iterable
from datetime import date
from typing import Self

import numpy as np
import polars
from numpy.typing import NDArray
from pandera.typing.polars import DataFrame
from scipy.interpolate import LinearNDInterpolator, interp1d

from vol_forge.constants import ASK, BID, MID, STRIKE, TTM
from vol_forge.forward.forward_curve import LinearForwardCurve
from vol_forge.market_data.schemas.constants import (
    COLUMN_ASK_IMPLIED_VOL,
    COLUMN_BID_IMPLIED_VOL,
    COLUMN_EXPIRY_DATE,
    COLUMN_FORWARD,
    COLUMN_MID_IMPLIED_VOL,
    COLUMN_STRIKE,
    COLUMN_TTM,
)
from vol_forge.market_data.schemas.implied_vols import ImpliedVolatilities
from vol_forge.slicer.slice import Slice
from vol_forge.surface.base_surface import BaseSurface


class LinearSurface(BaseSurface):
    def __init__(
        self,
        forward_curve: LinearForwardCurve,
        base_date: date,
        mid_interp: LinearNDInterpolator,
        bid_interp: LinearNDInterpolator,
        ask_interp: LinearNDInterpolator,
    ) -> None:
        super().__init__(forward_curve, base_date)
        self._mid_interp = mid_interp
        self._bid_interp = bid_interp
        self._ask_interp = ask_interp

    @classmethod
    def construct_from_table(
        cls, base_date: date, spot: float, implied_vols: DataFrame[ImpliedVolatilities]
    ) -> Self:
        forward_per_expiry = (
            implied_vols.select(COLUMN_EXPIRY_DATE, COLUMN_FORWARD)
            .unique(subset=COLUMN_EXPIRY_DATE)
            .iter_rows()
        )
        forward_curve = LinearForwardCurve(
            spine_date_value=list(forward_per_expiry),
            base_date=base_date,
            spot=spot,
        )
        points = implied_vols.select(COLUMN_TTM, COLUMN_STRIKE).to_numpy()
        if implied_vols[COLUMN_TTM].n_unique() > 1:
            mid_interp = LinearNDInterpolator(
                points, implied_vols[COLUMN_MID_IMPLIED_VOL].to_numpy()
            )
            bid_interp = LinearNDInterpolator(
                points, implied_vols[COLUMN_BID_IMPLIED_VOL].to_numpy()
            )
            ask_interp = LinearNDInterpolator(
                points, implied_vols[COLUMN_ASK_IMPLIED_VOL].to_numpy()
            )
        else:
            strikes = implied_vols[COLUMN_STRIKE].to_numpy()
            mid_interp = lambda t, k: interp1d(
                strikes,
                implied_vols[COLUMN_MID_IMPLIED_VOL].to_numpy(),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )(k)
            bid_interp = lambda t, k: interp1d(
                strikes,
                implied_vols[COLUMN_BID_IMPLIED_VOL].to_numpy(),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )(k)
            ask_interp = lambda t, k: interp1d(
                strikes,
                implied_vols[COLUMN_ASK_IMPLIED_VOL].to_numpy(),
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )(k)
        return cls(
            forward_curve=forward_curve,
            base_date=base_date,
            mid_interp=mid_interp,
            bid_interp=bid_interp,
            ask_interp=ask_interp,
        )

    @classmethod
    def construct_from_slices(
        cls, base_date: date, spot: float, slices: list[Slice]
    ) -> Self:
        forward_curve = LinearForwardCurve(
            spine_date_value=[(slice.expiry_date, slice.forward) for slice in slices],
            base_date=base_date,
            spot=spot,
        )

        mid_interp, bid_interp, ask_interp = cls._build_interpolators(slices)
        return cls(
            forward_curve=forward_curve,
            base_date=base_date,
            mid_interp=mid_interp,
            bid_interp=bid_interp,
            ask_interp=ask_interp,
        )

    @staticmethod
    def _build_interpolators(slices: list[Slice]) -> LinearNDInterpolator:
        points = []
        values_mid = []
        values_bid = []
        values_ask = []
        for s in slices:
            strikes = set(s.bid_implied_vols.keys()) & set(s.ask_implied_vols.keys())
            points.extend([(s.ttm, k) for k in strikes])
            values_mid.extend(
                [(s.bid_implied_vols[k] + s.ask_implied_vols[k]) / 2 for k in strikes]
            )
            values_bid.extend([s.bid_implied_vols[k] for k in strikes])
            values_ask.extend([s.ask_implied_vols[k] for k in strikes])
        return [
            LinearNDInterpolator(points, values_mid),
            LinearNDInterpolator(points, values_bid),
            LinearNDInterpolator(points, values_ask),
        ]

    def get_iv(
        self,
        maturities: Iterable[float],
        strikes: Iterable[float],
    ) -> NDArray[np.float64]:
        return self._mid_interp(maturities, strikes)

    def get_market_implied_vols(self) -> DataFrame[ImpliedVolatilities]:
        quotes_frame = polars.DataFrame(
            np.concat(
                [
                    self._mid_interp.points,
                    self._bid_interp.values,
                    self._ask_interp.values,
                    self._mid_interp.values,
                ],
                axis=1,
            )
        )
        quotes_frame.columns = [
            TTM,
            STRIKE,
            BID,
            ASK,
            MID,
        ]

        return quotes_frame
