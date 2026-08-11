import pandera.polars as pa


class OptionSliceSchema(pa.DataFrameModel):
    symbol: str
    strike: float
    right: str
    close: float = pa.Field(nullable=True)
    volume: int
    bid_size: int
    bid: float
    ask_size: int
    ask: float
    mid: float
    bid_ask_spread: float
