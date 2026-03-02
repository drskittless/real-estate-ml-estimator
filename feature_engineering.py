import pandas as pd
import numpy as np
from datetime import datetime


class FeatureEngineer:
    def __init__(self):
        self.current_year = datetime.now().year

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df["house_age"] = self.current_year - df["year_built"]

        df["price_per_sqft"] = df["price"] / df["sqft"]

        df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

        df["rooms_ratio"] = df["total_rooms"] / df["sqft"]

        # Area grouping (FinTech-style segmentation)
        df["area_group"] = pd.cut(
            df["sqft"],
            bins=[0, 1000, 2000, 3000, 10000],
            labels=["Small", "Medium", "Large", "Luxury"]
        )

        return df