import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_optimal_features_cu"]


def generate_optimal_features_cu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for optimisation of the 2nd and the 3d floating machines

    """

    logging.info("Adding process result features...")

    n = df.shape[0] - 1
    df.loc[:n, "Cu_2.1T_next"] = df.loc[1:, "Cu_2.1T"]
    df.loc[:n, "Cu_2.2T_next"] = df.loc[1:, "Cu_2.2T"]

    df.loc[:n, "Cu_3.1T_next"] = df.loc[1:, "Cu_3.1T"]
    df.loc[:n, "Cu_3.2T_next"] = df.loc[1:, "Cu_3.2T"]

    logging.info("Specifying optimisation conditions...")

    cond_1 = (df["Cu_2.1T_next"] - df["Cu_2.2T_next"] > 0) & (df["FM_2.2_A"] == 1)
    cond_2 = (df["Cu_2.1T_next"] - df["Cu_2.2T_next"] < 0) & (df["FM_2.1_A"] == 1)

    cond_3 = (df["Cu_3.1T_next"] - df["Cu_3.2T_next"] > 0) & (df["FM_3.2_A"] == 1)
    cond_4 = (df["Cu_3.1T_next"] - df["Cu_3.2T_next"] < 0) & (df["FM_3.1_A"] == 1)

    logging.info("Generating features according to selected conditions...")

    df.loc[cond_1, "Cu_2T_min"] = df.loc[cond_1, "Cu_2.2T_min"]
    df.loc[cond_1, "Cu_2T_max"] = df.loc[cond_1, "Cu_2.2T_max"]
    df.loc[cond_1, "Ni_2C"] = df.loc[cond_1, "Ni_2.2C"]
    df.loc[cond_1, "Cu_2C"] = df.loc[cond_1, "Cu_2.2C"]
    df.loc[cond_1, "Ni_2T"] = df.loc[cond_1, "Ni_2.2T"]
    df.loc[cond_1, "Cu_2T"] = df.loc[cond_1, "Cu_2.2T"]

    df.loc[cond_2, "Cu_2T_min"] = df.loc[cond_2, "Cu_2.1T_min"]
    df.loc[cond_2, "Cu_2T_max"] = df.loc[cond_2, "Cu_2.1T_max"]
    df.loc[cond_2, "Ni_2C"] = df.loc[cond_2, "Ni_2.1C"]
    df.loc[cond_2, "Cu_2C"] = df.loc[cond_2, "Cu_2.1C"]
    df.loc[cond_2, "Ni_2T"] = df.loc[cond_2, "Ni_2.1T"]
    df.loc[cond_2, "Cu_2T"] = df.loc[cond_2, "Cu_2.1T"]

    df.loc[cond_3, "Cu_3T_min"] = df.loc[cond_3, "Cu_3.2T_min"]
    df.loc[cond_3, "Cu_3T_max"] = df.loc[cond_3, "Cu_3.2T_min"]
    df.loc[cond_3, "Ni_3C"] = df.loc[cond_3, "Ni_3.2C"]
    df.loc[cond_3, "Cu_3C"] = df.loc[cond_3, "Cu_3.2C"]
    df.loc[cond_3, "Ni_3T"] = df.loc[cond_3, "Ni_3.2T"]
    df.loc[cond_3, "Cu_3T"] = df.loc[cond_3, "Cu_3.2T"]

    df.loc[cond_4, "Cu_3T_min"] = df.loc[cond_4, "Cu_3.1T_min"]
    df.loc[cond_4, "Cu_3T_max"] = df.loc[cond_4, "Cu_3.1T_max"]
    df.loc[cond_4, "Ni_3C"] = df.loc[cond_2, "Ni_3.1C"]
    df.loc[cond_4, "Cu_3C"] = df.loc[cond_2, "Cu_3.1C"]
    df.loc[cond_4, "Ni_3T"] = df.loc[cond_2, "Ni_3.1T"]
    df.loc[cond_4, "Cu_3T"] = df.loc[cond_2, "Cu_3.1T"]

    return df
