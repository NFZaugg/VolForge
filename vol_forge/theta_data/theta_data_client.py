from thetadata.client import ThetaClient

from vol_forge.config.config import config


class ThetaDataClientFactory:
    _instance: ThetaClient | None = None

    @classmethod
    def get_instance(cls) -> ThetaClient:
        if cls._instance is None:
            if config.thetadata_api_key is None:
                raise ValueError(
                    "To initalize ThetaDataClient via create_instance method, please set THETADATA_API_KEY in .env file!"
                )
            cls._instance = ThetaClient(
                api_key=str(config.thetadata_api_key),
                dataframe_type="polars",
            )
        return cls._instance

    @classmethod
    def create_instance(cls) -> ThetaClient:
        return cls.get_instance()
