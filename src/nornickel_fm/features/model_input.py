import logging
from typing import Optional, Tuple

import pandas as pd

from . import CONFIG

logger = logging.getLogger(__name__)

__all__ = ["make_dataset_ore"]


def make_dataset_ore(
    df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset for modeling flotation process of the 1st floating machine

    """
    if config is None:
        config = CONFIG

    fm_1_valid_ind = df[
        df[config["fm_1_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_1_target_cols"])
    ].index

    X_fm1 = df.loc[fm_1_valid_ind, config["fm_1_feature_cols"]]
    y_fm_1 = df.loc[fm_1_valid_ind, config["fm_1_target_cols"]]

    return X_fm1, y_fm_1
