from datetime import date

import numpy as np


class LinearForwardCurve:
    def __init__(
        self, spine_date_value: list[tuple[date, float]], base_date: date, spot: float
    ):
        self.spine_date_value = spine_date_value
        self.base_date = base_date

        spine_maturities = [0] + [
            (d - base_date).days / 365 for d, _ in spine_date_value
        ]
        spine_values = [spot] + [v for _, v in spine_date_value]

        self._interpolator = lambda x: np.interp(
            x, spine_maturities, spine_values, right=spine_values[-1]
        )

    def get_forward_from_date(self, forward_date: date) -> float:
        return self.get_forward(self.get_ttm(forward_date))

    def get_ttm(self, forward_date: date) -> float:
        return (forward_date - self.base_date).days / 365

    def get_forward(self, time_to_forward: float) -> float:
        return self._interpolator(time_to_forward)
