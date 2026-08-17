from datetime import date

import polars
from pandera.typing.polars import DataFrame

from vol_forge.market_data.schemas.implied_vols import ImpliedVolatilities
from vol_forge.slicer.slice import Slice
from vol_forge.slicer.slicer import Slicer
from vol_forge.surface.linear_surface import LinearSurface
from vol_forge.theta_data.options_theta_data_fetcher import OptionsThetaDataFetcher
from vol_forge.theta_data.theta_data_cleaner import ThetaDataCleaner
from vol_forge.visualization.slice_plotter import SlicePlotter
from vol_forge.visualization.surface_plotter import SurfacePlotter


class VolForgeClient:
    def __init__(self, liquidity_threshold: float = 1.0, dark_theme: bool = True):

        self._fetcher = OptionsThetaDataFetcher.create_instance()
        self._cleaner = ThetaDataCleaner(liquidity_threshold=liquidity_threshold)
        self._slice_plotter = SlicePlotter(dark_theme=dark_theme)
        self._surface_plotter = SurfacePlotter(dark_theme=dark_theme)

    def fetch_ivs_for_date(
        self, symbol: str, value_date: date, expiration_date: date
    ) -> DataFrame[ImpliedVolatilities]:
        slice = self._fetch_iv_slice(symbol, value_date, expiration_date)
        return slice.to_polars()

    def fetch_ivs_all_expiries_for_date(
        self, symbol: str, value_date: date, min_date: date, max_date: date
    ) -> DataFrame[ImpliedVolatilities]:
        slices = self._fetch_iv_slices_many_expirations(
            symbol, value_date, min_date, max_date
        )
        return polars.concat([slice.to_polars() for slice in slices])

    def plot_single_expiry(
        self, symbol: str, value_date: date, expiration_date: date
    ) -> DataFrame[ImpliedVolatilities]:
        slice = self._fetch_iv_slice(symbol, value_date, expiration_date)
        return self._slice_plotter.plot(symbol, slice)

    def plot_surface(
        self,
        symbol: str,
        value_date: date,
        min_expiration_date: date,
        max_expiration_date: date,
    ) -> DataFrame[ImpliedVolatilities]:
        slices = self._fetch_iv_slices_many_expirations(
            symbol, value_date, min_expiration_date, max_expiration_date
        )
        spot_price = self._fetcher.fetch_eod_close(symbol=symbol, value_date=value_date)
        surface = LinearSurface.construct_from_slices(value_date, spot_price, slices)
        return self._surface_plotter.plot(symbol, surface)

    # Private Methods
    def _fetch_iv_slice(
        self, symbol: str, value_date: date, expiration_date: date
    ) -> Slice | None:
        data = self._fetcher.fetch_options_eod_history(
            symbol=symbol,
            start_date=value_date,
            end_date=value_date,
            expiration=expiration_date,
        )
        cleaned_data = self._cleaner.clean(data)
        slice = Slicer(base_date=value_date).construct_slice(
            cleaned_data, expiry_date=expiration_date
        )
        return slice

    def _fetch_iv_slices_many_expirations(
        self, symbol: str, value_date: date, min_date: date, max_date: date
    ) -> list[Slice]:
        data = self._fetcher.fetch_options_eod_for_all_expiries(
            symbol=symbol,
            base_date=value_date,
            min_date=min_date,
            max_date=max_date,
        )
        cleaned_data = self._cleaner.clean(data)
        slices = Slicer(base_date=value_date).construct_slices(cleaned_data)
        return slices
