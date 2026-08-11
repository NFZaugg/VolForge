from datetime import date, datetime

from pandera.polars import DataFrameModel


class ImpliedVolatilities(DataFrameModel):
    underlying_ticker: str
    value_datetime: datetime
    forward: float
    ttm: float
    expiry_date: date
    strike: float
    mid_implied_vol: float
    bid_implied_vol: float
    ask_implied_vol: float
