import datetime
import logging
from typing import Self

import polars
from pandera.typing.polars import DataFrame
from thetadata.client import ThetaClient
from thetadata.errors import NoDataFoundError

from theta_data.schemas.theta_data_schema import TDHistoricalOptionData
from theta_data.theta_data_client import ThetaDataClientFactory

logger = logging.getLogger(__name__)


class OptionsThetaDataFetcher:
    def __init__(self, theta_client: ThetaClient) -> None:
        self.theta_client = theta_client

    @classmethod
    def create_instance(cls) -> Self:
        theta_client = ThetaDataClientFactory.create_instance()

        return cls(theta_client)

    def fetch_options_eod_history(
        self,
        *,
        start_date: datetime.date,
        end_date: datetime.date,
        symbol: str,
        expiration: datetime.date,
    ) -> DataFrame[TDHistoricalOptionData]:
        logger.info(f"Fetching {expiration}")
        try:
            raw_data = self.theta_client.option_history_eod(
                start_date, end_date, symbol, expiration
            )
        except NoDataFoundError:
            logger.warning(f"no Data for {symbol} with expiration {expiration}")
            return TDHistoricalOptionData.empty()
        return TDHistoricalOptionData.validate(raw_data)

    def get_expiries_for_symbol(
        self,
        *,
        symbol: str,
        base_date: datetime.date,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
    ) -> list[datetime.date]:
        expiries = (
            self.theta_client.option_list_expirations([symbol])
            .with_columns(polars.col("expiration").cast(polars.Date))["expiration"]
            .to_list()
        )
        max_date = max_date if max_date is not None else max(expiries)
        min_date = min_date if min_date is not None else base_date
        return [e for e in expiries if min_date <= e <= max_date]

    def fetch_options_eod_for_all_expiries(
        self,
        *,
        symbol: str,
        base_date: datetime.date,
        min_date: datetime.date | None = None,
        max_date: datetime.date | None = None,
    ) -> DataFrame[TDHistoricalOptionData]:
        expiries = self.get_expiries_for_symbol(
            symbol=symbol, base_date=base_date, min_date=min_date, max_date=max_date
        )
        quotes = {
            expiry_date: data
            for expiry_date in expiries
            if not (
                data := self.fetch_options_eod_history(
                    start_date=base_date,
                    end_date=base_date,
                    symbol=symbol,
                    expiration=expiry_date,
                )
            ).is_empty()
        }
        if len(quotes) > 0:
            return polars.concat(quotes.values())
        else:
            return TDHistoricalOptionData.empty()

    def fetch_eod_close(self, *, symbol: str, value_date: datetime.date) -> float:
        return self.theta_client.stock_history_eod(symbol, value_date, value_date)[
            "close"
        ][0]
