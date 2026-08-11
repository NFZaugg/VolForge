from datetime import date

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def get_trading_days(start_date: date, end_date: date) -> list[date]:
    return pd.date_range(start_date, end_date, freq=us_bd).date
