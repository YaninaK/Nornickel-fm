import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_optimal_features_ore"]


def generate_optimal_features_ore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for optimisation of the 1st floating machine.

    """
    logging.info("Adding process result features...")

    n = df.shape[0] - 1
    df.loc[:n, "Ni_1.1C_next"] = df.loc[1:, "Ni_1.1C"]
    df.loc[:n, "Ni_1.2C_next"] = df.loc[1:, "Ni_1.2C"]

    df.loc[:n, "Cu_1.1C_next"] = df.loc[1:, "Cu_1.1C"]
    df.loc[:n, "Cu_1.2C_next"] = df.loc[1:, "Cu_1.2C"]

    logging.info("Specifying optimisation conditions...")

    cond_1 = (df["Ni_1.1C_next"] - df["Ni_1.2C_next"] > 0) & (df["FM_1.1_A"] == 1)
    cond_2 = (df["Ni_1.1C_next"] - df["Ni_1.2C_next"] < 0) & (df["FM_1.2_A"] == 1)

    cond_3 = (df["Cu_1.1C_next"] - df["Cu_1.2C_next"] > 0) & (df["FM_1.1_A"] == 1)
    cond_4 = (df["Cu_1.1C_next"] - df["Cu_1.2C_next"] < 0) & (df["FM_1.2_A"] == 1)

    logging.info("Generating features according to selected conditions...")

    df.loc[cond_1, "Ni_1C_min"] = df.loc[cond_1, "Ni_1.1C_min"]
    df.loc[cond_1, "Ni_1C_max"] = df.loc[cond_1, "Ni_1.1C_max"]
    df.loc[cond_1, "Ni_1C"] = df.loc[cond_1, "Ni_1.1C"]

    df.loc[cond_2, "Ni_1C_min"] = df.loc[cond_2, "Ni_1.2C_min"]
    df.loc[cond_2, "Ni_1C_max"] = df.loc[cond_2, "Ni_1.2C_max"]
    df.loc[cond_2, "Ni_1C"] = df.loc[cond_1, "Ni_1.2C"]

    df.loc[cond_3, "Cu_1C_min"] = df.loc[cond_3, "Cu_1.1C_min"]
    df.loc[cond_3, "Cu_1C_max"] = df.loc[cond_3, "Cu_1.1C_min"]
    df.loc[cond_3, "Cu_1C"] = df.loc[cond_3, "Cu_1.1C"]

    df.loc[cond_4, "Cu_1C_min"] = df.loc[cond_4, "Cu_1.2C_min"]
    df.loc[cond_4, "Cu_1C_max"] = df.loc[cond_4, "Cu_1.2C_max"]
    df.loc[cond_4, "Cu_1C"] = df.loc[cond_4, "Cu_1.2C"]

    return df
