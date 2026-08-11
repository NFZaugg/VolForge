import logging
from datetime import date

from matplotlib import pyplot as plt

from slicer.slicer import Slicer
from surface.linear_surface import LinearSurface
from theta_data.options_theta_data_fetcher import OptionsThetaDataFetcher
from theta_data.theta_data_cleaner import ThetaDataCleaner
from visualization.slice_plotter import SlicePlotter
from visualization.surface_plotter import SurfacePlotter

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
if __name__ == "__main__":
    base_date = date(2026, 7, 24)
    min_date = date(2026, 8, 7)
    max_date = date(2026, 9, 30)
    symbol = "AAPL"

    # Init Fetcher and data cleaner
    fetcher = OptionsThetaDataFetcher.create_instance()
    cleaner = ThetaDataCleaner(liquidity_threshold=1)

    # Fetch and clean data
    spot_price = fetcher.fetch_eod_close(symbol=symbol, value_date=base_date)
    options_data = fetcher.fetch_options_eod_for_all_expiries(
        symbol=symbol, base_date=base_date, max_date=max_date, min_date=min_date
    )

    cleaned_data_per_expiry = cleaner.clean(options_data)

    # Construct surface
    slices = Slicer(
        base_date=base_date,
    ).construct_slices(cleaned_data_per_expiry)
    for slice in slices:
        SlicePlotter(dark_theme=True).plot(symbol, slice)
        plt.show()

    surface = LinearSurface.construct_from_slices(
        base_date, spot=spot_price, slices=slices
    )

    # Plot Surface
    fig, ax = SurfacePlotter(dark_theme=True).plot(
        underlying=symbol,
        surface=surface,
    )

    plt.savefig("surface.png", dpi=200)
