import datetime
import logging
from typing import Self

from pandera.typing.polars import DataFrame
from thetadata.client import ThetaClient
from thetadata.errors import NoDataFoundError

from vol_forge.theta_data.schemas.theta_data_schema import TDHistoricalEquityData
from vol_forge.theta_data.theta_data_client import ThetaDataClientFactory

logger = logging.getLogger(__name__)


class EquitiesThetaDataFetcher:
    def __init__(self, theta_client: ThetaClient) -> None:
        self.theta_client = theta_client

    @classmethod
    def create_instance(cls) -> Self:
        theta_client = ThetaDataClientFactory.create_instance()

        return cls(theta_client)

    def fetch_eod_history(
        self,
        *,
        start_date: datetime.date,
        end_date: datetime.date,
        symbol: str,
    ) -> DataFrame[TDHistoricalEquityData]:
        logger.info(f"Fetching {symbol}")
        try:
            return TDHistoricalEquityData.validate(
                self.theta_client.stock_history_eod(
                    start_date=start_date, end_date=end_date, symbol=symbol
                )
            )
        except NoDataFoundError:
            logger.warning(f"no Data for {symbol}")
            return TDHistoricalEquityData.empty()
