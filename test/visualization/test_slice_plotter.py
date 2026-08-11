from datetime import date
from types import SimpleNamespace

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from slicer.slice import Slice
from visualization.slice_plotter import SlicePlotter


class TestSlicePlotter:
    def _create_mock_slice(self) -> Slice:

        return SimpleNamespace(
            bid_implied_vols={20: 0.5, 30: 0.6, 40: 0.7},
            ask_implied_vols={20: 0.55, 30: 0.65, 40: 0.75},
            base_date=date(2026, 1, 1),
            expiry_date=date(2026, 6, 1),
            ttm=0.5,
            forward=30,
        )

    def test_slice_plotter(self):
        # Given:
        slice = self._create_mock_slice()
        # When:
        fig, ax = SlicePlotter(dark_theme=True).plot("ABC", slice)
        # Then:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
