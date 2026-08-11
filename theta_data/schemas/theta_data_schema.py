import pandera.polars as pa
from polars import Datetime


class TDHistoricalOptionData(pa.DataFrameModel):
    symbol: str
    expiration: str
    strike: float
    right: str
    created: Datetime(time_unit="ms", time_zone="America/New_York")  # pyright: ignore[reportInvalidTypeForm]
    last_trade: Datetime(time_unit="ms", time_zone="America/New_York") = (  # pyright: ignore[reportInvalidTypeForm]
        pa.Field(nullable=True)
    )
    open: float = pa.Field(nullable=True)
    high: float = pa.Field(nullable=True)
    low: float = pa.Field(nullable=True)
    close: float = pa.Field(nullable=True)
    volume: int
    count: int
    bid_size: int
    bid_exchange: int = pa.Field(nullable=True)
    bid: float = pa.Field(nullable=True)
    bid_condition: int = pa.Field(nullable=True)
    ask_size: int
    ask_exchange: int = pa.Field(nullable=True)
    ask: float = pa.Field(nullable=True)
    ask_condition: int = pa.Field(nullable=True)


class TDHistoricalEquityData(pa.DataFrameModel):
    created: Datetime(time_unit="ms", time_zone="America/New_York")  # pyright: ignore[reportInvalidTypeForm]
    last_trade: Datetime(time_unit="ms", time_zone="America/New_York")  # pyright: ignore[reportInvalidTypeForm]

    open: float = pa.Field(nullable=True)
    high: float = pa.Field(nullable=True)
    low: float = pa.Field(nullable=True)
    close: float = pa.Field(nullable=True)

    volume: int
    count: int

    bid_size: int
    bid_exchange: int = pa.Field(nullable=True)
    bid: float = pa.Field(nullable=True)
    bid_condition: int = pa.Field(nullable=True)

    ask_size: int
    ask_exchange: int = pa.Field(nullable=True)
    ask: float = pa.Field(nullable=True)
    ask_condition: int = pa.Field(nullable=True)
