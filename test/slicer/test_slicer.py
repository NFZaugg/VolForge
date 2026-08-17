from datetime import date

import numpy as np
import polars

from vol_forge.constants import EXPIRATION
from vol_forge.slicer.slicer import Slicer
from vol_forge.theta_data.theta_data_cleaner import ThetaDataCleaner


class TestSlicer:
    def test_slicer_real_data(self):

        raw_data = polars.read_csv("test/theta_data/fixtures/thetadata_raw.csv")
        d = date(2026, 6, 12)
        cleaned_data = ThetaDataCleaner(liquidity_threshold=0.5).clean(raw_data)
        option_slice_data_per_expiry = {
            _date_key[0]: slice_data
            for _date_key, slice_data in cleaned_data.partition_by(
                EXPIRATION, as_dict=True
            ).items()
        }
        for expiry_date, data in option_slice_data_per_expiry.items():
            slice = Slicer(d).construct_slice(data, expiry_date=expiry_date)
            np.testing.assert_almost_equal(slice.forward, 22.0924383634569)
            assert len(slice.bid_implied_vols) == 41
            assert len(slice.ask_implied_vols) == 60

            assert len(slice.bids) == 60
            assert len(slice.asks) == 60
            assert len(slice.mids) == 60
            assert slice.base_date == d
            assert slice.expiry_date == expiry_date
