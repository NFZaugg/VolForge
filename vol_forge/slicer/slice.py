from datetime import date

import polars
from pandera.typing.polars import DataFrame
from pydantic import BaseModel

from vol_forge.market_data.schemas.constants import (
    COLUMN_ASK_IMPLIED_VOL,
    COLUMN_BASE_DATE,
    COLUMN_BID_IMPLIED_VOL,
    COLUMN_EXPIRY_DATE,
    COLUMN_FORWARD,
    COLUMN_MID_IMPLIED_VOL,
    COLUMN_STRIKE,
    COLUMN_TTM,
    COLUMN_UNDERLYING_TICKER,
)
from vol_forge.market_data.schemas.implied_vols import ImpliedVolatilities
from vol_forge.tiny_types import Strike, TimeToExpiry


class Slice(BaseModel):
    bid_implied_vols: dict[Strike, float]
    ask_implied_vols: dict[Strike, float]
    bids: dict[Strike, float]
    asks: dict[Strike, float]
    mids: dict[Strike, float]

    base_date: date
    expiry_date: date

    ttm: TimeToExpiry
    forward: float
    rate: float
    underlying_ticker: str

    def to_polars(self) -> DataFrame[ImpliedVolatilities]:
        all_strikes = sorted(
            set(self.bid_implied_vols.keys()) | set(self.ask_implied_vols.keys())
        )

        return ImpliedVolatilities.validate(
            polars.DataFrame(
                {
                    COLUMN_UNDERLYING_TICKER: self.underlying_ticker,
                    COLUMN_BASE_DATE: self.base_date,
                    COLUMN_EXPIRY_DATE: self.expiry_date,
                    COLUMN_TTM: float(self.ttm),
                    COLUMN_FORWARD: self.forward,
                    COLUMN_STRIKE: all_strikes,
                    COLUMN_BID_IMPLIED_VOL: [
                        self.bid_implied_vols.get(k) for k in all_strikes
                    ],
                    COLUMN_ASK_IMPLIED_VOL: [
                        self.ask_implied_vols.get(k) for k in all_strikes
                    ],
                }
            )
            .drop_nulls(subset=[COLUMN_BID_IMPLIED_VOL, COLUMN_ASK_IMPLIED_VOL])
            .with_columns(
                (
                    0.5
                    * (
                        polars.col(COLUMN_BID_IMPLIED_VOL)
                        + polars.col(COLUMN_ASK_IMPLIED_VOL)
                    )
                ).alias(COLUMN_MID_IMPLIED_VOL)
            )
        )
