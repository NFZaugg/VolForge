from datetime import date

from pydantic import BaseModel

from tiny_types import Strike, TimeToExpiry


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
