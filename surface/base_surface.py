from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import date

import numpy as np
from numpy.typing import NDArray

from forward.forward_curve import LinearForwardCurve


class BaseSurface(ABC):
    def __init__(self, forward_curve: LinearForwardCurve, base_date: date):
        self.forward_curve = forward_curve
        self.base_date = base_date

    @abstractmethod
    def get_iv(
        self, maturities: Iterable[float], strikes: Iterable[float]
    ) -> NDArray[np.float64]:
        pass
