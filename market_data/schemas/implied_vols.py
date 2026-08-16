from datetime import date

from pandera.polars import DataFrameModel


class ImpliedVolatilities(DataFrameModel):
    underlying_ticker: str
    forward: float
    ttm: float
    expiry_date: date
    base_date: date
    strike: float
    mid_implied_vol: float
    bid_implied_vol: float
    ask_implied_vol: float
