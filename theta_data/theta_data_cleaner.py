from datetime import date

import polars
from pandera.typing.polars import DataFrame

from constants import (
    ASK,
    ASK_SIZE,
    BID,
    BID_ASK_SPREAD,
    BID_SIZE,
    CLOSE,
    EXPIRATION,
    MID,
    RIGHT,
    STRIKE,
    SYMBOL,
    VOLUME,
)
from slicer.schemas.slice_data import OptionSliceSchema
from theta_data.schemas.theta_data_schema import TDHistoricalOptionData

RELEVANT_COLUMNS = [
    SYMBOL,
    EXPIRATION,
    STRIKE,
    RIGHT,
    CLOSE,
    VOLUME,
    BID_SIZE,
    BID,
    ASK_SIZE,
    ASK,
]


class ThetaDataCleaner:
    def __init__(
        self,
        liquidity_threshold: float,
    ):
        self.liquidity_threshold = liquidity_threshold

    def clean(
        self, raw_data: DataFrame[TDHistoricalOptionData]
    ) -> dict[date, DataFrame[OptionSliceSchema]]:

        cleaned_quotes = (
            raw_data.select(RELEVANT_COLUMNS)
            .with_columns(polars.col(EXPIRATION).cast(polars.Date))
            .filter(
                (polars.col(BID) > 0) & (polars.col(ASK) > 0),
                polars.col(BID) <= polars.col(ASK),
            )
            .with_columns(((polars.col(BID) + polars.col(ASK)) / 2).alias(MID))
            .with_columns(
                ((polars.col(ASK) - polars.col(BID)) / polars.col(MID)).alias(
                    BID_ASK_SPREAD
                ),
            )
        )
        cleaned_quotes = cleaned_quotes.filter(
            polars.col(BID_ASK_SPREAD) < self.liquidity_threshold
        )
        return cleaned_quotes
