from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Configuration(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    thetadata_api_key: str | None  = None
    debug: bool = False


@lru_cache
def _get_config():
    return Configuration()


config = _get_config()