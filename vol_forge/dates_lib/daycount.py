from datetime import date


def get_daycount(*, start_date: date, end_date: date) -> float:
    ttm = (end_date - start_date).days / 365
    return ttm
