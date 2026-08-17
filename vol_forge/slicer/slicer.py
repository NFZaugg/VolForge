import logging
from datetime import date

import numpy as np
import pandas as pd
import polars
from numpy.typing import NDArray
from pandera.typing.polars import DataFrame

from vol_forge.black_scholes_lib.black.pricing import get_implied_vol_black
from vol_forge.constants import (
    ASK,
    ASK_DIFF,
    BID,
    BID_ASK_SPREAD,
    BID_DIFF,
    CALL,
    EXPIRATION,
    MID,
    PUT,
    RIGHT,
    STRIKE,
    SYMBOL,
)
from vol_forge.dates_lib.daycount import get_daycount
from vol_forge.slicer.schemas.slice_data import OptionSliceSchema
from vol_forge.slicer.slice import Slice
from vol_forge.tiny_types import Strike, TimeToExpiry

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_RATE = 0.04
DEFAULT_RATE_BOUNDS = (0, 0.1)


class Slicer:
    def __init__(
        self,
        base_date: date,
        fallback_rate: float = DEFAULT_FALLBACK_RATE,
        rate_bounds: tuple[float] = DEFAULT_RATE_BOUNDS,
    ):
        self.base_date = base_date
        self.fallback_rate = fallback_rate
        self.rate_bounds = rate_bounds

    def _get_expiry(self, quotes: pd.DataFrame) -> date:
        if len(expiry_dates := quotes[EXPIRATION].unique()) > 1:
            raise ValueError("Frame Contains multiple expiries")
        return date.fromisoformat(expiry_dates[0])

    def _get_forward(
        self, ttm: float, strikes_with_both_quotes: DataFrame[OptionSliceSchema]
    ) -> tuple[float, float]:

        f, rate = self._compute_implied_forward_regression(
            strikes_with_both_quotes, ttm
        )
        return f, rate

    def _convert_to_call(
        self,
        cleaned_quotes: DataFrame[OptionSliceSchema],
        forward: float,
        ttm: float,
        rate,
    ) -> polars.DataFrame:
        discount_factor = np.exp(-rate * ttm)
        cleaned_quotes_converted = cleaned_quotes.with_columns(
            polars.when(polars.col(RIGHT) == PUT)
            .then(
                polars.max_horizontal(
                    (forward - polars.col(STRIKE)) * discount_factor + polars.col(ASK),
                    0,
                )
            )
            .otherwise(polars.col(ASK))
            .alias(ASK),
            polars.when(polars.col(RIGHT) == PUT)
            .then(
                polars.max_horizontal(
                    (forward - polars.col(STRIKE)) * discount_factor + polars.col(BID),
                    0,
                )
            )
            .otherwise(polars.col(BID))
            .alias(BID),
        )
        cleaned_quotes_converted = cleaned_quotes_converted.with_columns(
            ((polars.col(BID) + polars.col(ASK)) / 2).alias(MID)
        )
        cleaned_quotes_converted = cleaned_quotes_converted.with_columns(
            ((polars.col(ASK) - polars.col(BID)) / polars.col(MID)).alias(
                BID_ASK_SPREAD
            )
        )

        reduced = cleaned_quotes_converted.group_by(STRIKE).agg(
            polars.all().gather(polars.col(BID_ASK_SPREAD).arg_min()).first()
        )
        return reduced.sort(STRIKE)

    def construct_slice(
        self,
        option_slice_data: DataFrame[OptionSliceSchema],
        expiry_date: date,
    ) -> Slice | None:
        symbol = option_slice_data[SYMBOL].head(1).item()
        ttm = get_daycount(start_date=self.base_date, end_date=expiry_date)
        if ttm < 1 / 365:
            logger.warning(
                f"Unable to construct slice for {expiry_date} as options are maturing within less than a day. Skipping maturity date"
            )
            return None

        if (
            len(
                strikes_with_both_quotes := option_slice_data.filter(
                    polars.len().over(STRIKE) == 2
                )
            )
            < 1
        ):
            logger.warning(
                f"Unable to construct slice for {expiry_date} as there are no quotes with both put and calls. Skipping maturity date"
            )
            return None
        forward, rate = self._get_forward(ttm, strikes_with_both_quotes)

        call_quotes = self._convert_to_call(option_slice_data, forward, ttm, rate)

        strikes = call_quotes[STRIKE].to_numpy()

        ask_implied_vols = self._compute_ivs(
            forward, ttm, call_quotes[ASK].to_numpy(), strikes, rate
        )
        bid_implied_vols = self._compute_ivs(
            forward, ttm, call_quotes[BID].to_numpy(), strikes, rate
        )

        bids, asks, mids = self._extract_quote_dicts(call_quotes)
        return Slice(
            bid_implied_vols=bid_implied_vols,
            bids=bids,
            ask_implied_vols=ask_implied_vols,
            asks=asks,
            mids=mids,
            base_date=self.base_date,
            expiry_date=expiry_date,
            forward=forward,
            ttm=ttm,
            rate=rate,
            underlying_ticker=symbol,
        )

    def construct_slices(
        self, option_slice_data: DataFrame[OptionSliceSchema]
    ) -> list[Slice]:
        option_slice_data_per_expiry = {
            _date_key[0]: OptionSliceSchema.validate(slice_data)
            for _date_key, slice_data in option_slice_data.partition_by(
                EXPIRATION, as_dict=True
            ).items()
        }
        return [
            slice
            for expiry_date, option_slice_data in option_slice_data_per_expiry.items()
            if (
                slice := self.construct_slice(
                    option_slice_data=option_slice_data, expiry_date=expiry_date
                )
            )
            is not None
        ]

    def _extract_quote_dicts(
        self,
        call_quotes: polars.DataFrame,
    ) -> tuple[dict[Strike, float], dict[Strike, float], dict[Strike, float]]:
        bids = dict(call_quotes[STRIKE, BID].iter_rows())
        asks = dict(call_quotes[STRIKE, ASK].iter_rows())
        mids = dict(call_quotes[STRIKE, MID].iter_rows())
        return bids, asks, mids

    def _compute_ivs(
        self,
        forward: float,
        ttm: TimeToExpiry,
        call_quotes: NDArray[np.float64],
        strikes: list[Strike],
        r: float,
    ) -> dict[Strike, float]:
        implied_vols = get_implied_vol_black(call_quotes, forward, strikes, ttm, r)

        implied_vols = {
            k: iv
            for k, iv in zip(strikes, implied_vols)
            if not np.isnan(iv) and iv > 1e-10
        }
        return implied_vols

    def _compute_implied_forward_regression(
        self, strikes_with_both_quotes: polars.DataFrame, ttm: float
    ) -> tuple[float, float]:
        implied_forwards_per_strike = (
            strikes_with_both_quotes.group_by(STRIKE)
            .agg(
                (
                    -polars.col(BID).filter(polars.col(RIGHT) == CALL).first()
                    + polars.col(ASK).filter(polars.col(RIGHT) == PUT).first()
                ).alias(BID_DIFF),
                (
                    -polars.col(ASK).filter(polars.col(RIGHT) == CALL).first()
                    + polars.col(BID).filter(polars.col(RIGHT) == PUT).first()
                ).alias(ASK_DIFF),
            )
            .sort(STRIKE)
        )
        strikes = implied_forwards_per_strike[STRIKE].to_numpy()

        bid_diffs = implied_forwards_per_strike[BID_DIFF].to_numpy()
        ask_diffs = implied_forwards_per_strike[ASK_DIFF].to_numpy()
        mid_diffs = (bid_diffs + ask_diffs) / 2

        discount_factor, negative_discounted_forward = np.polyfit(strikes, mid_diffs, 1)

        if (rate := -np.log(discount_factor) / ttm) > self.rate_bounds[
            1
        ] or rate < self.rate_bounds[0]:
            rate = self.fallback_rate
            tightest_strike = np.argmin(bid_diffs - ask_diffs)
            discount_factor = np.exp(-rate * ttm)
            forward = (
                -(mid_diffs - np.exp(-rate * ttm) * strikes)[tightest_strike]
                / discount_factor
            )
        else:
            forward = -negative_discounted_forward / discount_factor

        return forward, rate
