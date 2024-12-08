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


def make_dataset_cu(
    df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset for modeling flotation process of the 2nd and the 3d floating machine

    """
    if config is None:
        config = CONFIG

    fm_2_valid_ind = df[
        df[config["fm_2_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_2_target_cols"])
    ].index
    fm_3_valid_ind = df[
        df[config["fm_3_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_3_target_cols"])
    ].index

    X_fm_2 = df.loc[fm_2_valid_ind, config["fm_2_feature_cols"]]
    X_fm_2["fm"] = 0
    y_fm_2 = df.loc[fm_2_valid_ind, config["fm_2_target_cols"]]

    X_fm_3 = df.loc[fm_3_valid_ind, config["fm_3_feature_cols"]]
    X_fm_3["fm"] = 1
    y_fm_3 = df.loc[fm_3_valid_ind, config["fm_3_target_cols"]]

    X_fm_2.columns = config["fm_2_3_feature_cols"]
    X_fm_3.columns = config["fm_2_3_feature_cols"]
    y_fm_2.columns = config["fm_2_3_target_cols"]
    y_fm_3.columns = config["fm_2_3_target_cols"]

    X_fm_2_3 = pd.concat([X_fm_2, X_fm_3], axis=0)
    y_fm_2_3 = pd.concat([y_fm_2, y_fm_3], axis=0)

    return X_fm_2_3, y_fm_2_3


def make_dataset_ni(
    df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset for modeling flotation process of the 4th, 5th and 6th floating machine

    """
    if config is None:
        config = CONFIG

    fm_4_valid_ind = df[
        df[config["fm_4_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_4_target_cols"])
    ].index
    fm_5_valid_ind = df[
        df[config["fm_5_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_5_target_cols"])
    ].index
    fm_6_valid_ind = df[
        df[config["fm_6_target_cols"]].notnull().sum(axis=1)
        == len(config["fm_6_target_cols"])
    ].index

    X_fm_4 = df.loc[fm_4_valid_ind, config["fm_4_feature_cols"]]
    X_fm_4["fm"] = 0
    y_fm_4 = df.loc[fm_4_valid_ind, config["fm_4_target_cols"]]

    X_fm_5 = df.loc[fm_5_valid_ind, config["fm_5_feature_cols"]]
    X_fm_5["Cu_5F"] = 0
    X_fm_5["fm"] = 1
    
    y_fm_5 = df.loc[fm_5_valid_ind, config["fm_5_target_cols"]]

    X_fm_6 = df.loc[fm_6_valid_ind, config["fm_6_feature_cols"]]
    X_fm_6["Cu_6F"] = 0
    X_fm_6["fm"] = 2
    y_fm_6 = df.loc[fm_6_valid_ind, config["fm_6_target_cols"]]

    X_fm_4.columns = config["fm_4_5_6_feature_cols"]
    X_fm_5.columns = config["fm_4_5_6_feature_cols"]
    X_fm_6.columns = config["fm_4_5_6_feature_cols"]
    X_fm_4_5_6 = pd.concat([X_fm_4, X_fm_5, X_fm_6], axis=0)

    y_fm_4.columns = config["fm_4_5_6_target_cols"]
    y_fm_5.columns = config["fm_4_5_6_target_cols"]
    y_fm_6.columns = config["fm_4_5_6_target_cols"]
    y_fm_4_5_6 = pd.concat([y_fm_4, y_fm_5, y_fm_6], axis=0)

    return X_fm_4_5_6, y_fm_4_5_6
