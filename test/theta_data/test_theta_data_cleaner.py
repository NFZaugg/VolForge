from datetime import date

import polars

from vol_forge.constants import BID_ASK_SPREAD, EXPIRATION
from vol_forge.theta_data.theta_data_cleaner import ThetaDataCleaner


class TestThetaDataCleaner:
    def test_clean_real_data_ok(self):

        raw_data = polars.read_csv("test/theta_data/fixtures/thetadata_raw.csv")

        cleaned_data_per_expiry = ThetaDataCleaner(liquidity_threshold=0.5).clean(
            raw_data
        )

        assert len(cleaned_data_per_expiry) == 87

        assert cleaned_data_per_expiry[EXPIRATION].head(1).item() == date(2026, 8, 19)
        assert (cleaned_data_per_expiry[BID_ASK_SPREAD] < 0.5).all()
