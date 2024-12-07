import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_optimal_features_ni"]


def generate_optimal_features_ni(df: pd.DataFrame) -> pd.DataFrame:

    logging.info("Adding process result features...")

    n = df.shape[0] - 1
    df.loc[:n, "Ni_4.1C_next"] = df.loc[1:, "Ni_4.1C"]
    df.loc[:n, "Ni_4.2C_next"] = df.loc[1:, "Ni_4.2C"]
    df.loc[:n, "Ni_4.1T_next"] = df.loc[1:, "Ni_4.1T"]
    df.loc[:n, "Ni_4.2T_next"] = df.loc[1:, "Ni_4.2T"]

    df.loc[:n, "Ni_5.1C_next"] = df.loc[1:, "Ni_5.1C"]
    df.loc[:n, "Ni_5.2C_next"] = df.loc[1:, "Ni_5.2C"]
    df.loc[:n, "Ni_5.1T_next"] = df.loc[1:, "Ni_5.1T"]
    df.loc[:n, "Ni_5.2T_next"] = df.loc[1:, "Ni_5.2T"]

    df.loc[:n, "Ni_6.1C_next"] = df.loc[1:, "Ni_6.1C"]
    df.loc[:n, "Ni_6.2C_next"] = df.loc[1:, "Ni_6.2C"]
    df.loc[:n, "Ni_6.1T_next"] = df.loc[1:, "Ni_6.1T"]
    df.loc[:n, "Ni_6.2T_next"] = df.loc[1:, "Ni_6.2T"]

    logging.info("Specifying optimisation conditions...")

    cond_1 = (df["Ni_4.1C_next"] - df["Ni_4.2C_next"] > 0) & (df["FM_4.1_A"] == 1)
    cond_2 = (df["Ni_4.1C_next"] - df["Ni_4.2C_next"] < 0) & (df["FM_4.2_A"] == 1)
    cond_3 = (df["Ni_4.1T_next"] - df["Ni_4.2T_next"] > 0) & (df["FM_4.2_A"] == 1)
    cond_4 = (df["Ni_4.1T_next"] - df["Ni_4.2T_next"] < 0) & (df["FM_4.1_A"] == 1)

    cond_5 = (df["Ni_5.1C_next"] - df["Ni_5.2C_next"] > 0) & (df["FM_5.1_A"] == 1)
    cond_6 = (df["Ni_5.1C_next"] - df["Ni_5.2C_next"] < 0) & (df["FM_5.2_A"] == 1)
    cond_7 = (df["Ni_5.1T_next"] - df["Ni_5.2T_next"] > 0) & (df["FM_5.2_A"] == 1)
    cond_8 = (df["Ni_5.1T_next"] - df["Ni_5.2T_next"] < 0) & (df["FM_5.1_A"] == 1)

    cond_9 = (df["Ni_6.1C_next"] - df["Ni_6.2C_next"] > 0) & (df["FM_6.1_A"] == 1)
    cond_10 = (df["Ni_6.1C_next"] - df["Ni_6.2C_next"] < 0) & (df["FM_6.2_A"] == 1)
    cond_11 = (df["Ni_6.1T_next"] - df["Ni_6.2T_next"] > 0) & (df["FM_6.2_A"] == 1)
    cond_12 = (df["Ni_6.1T_next"] - df["Ni_6.2T_next"] < 0) & (df["FM_6.1_A"] == 1)

    logging.info("Generating features according to selected conditions...")

    df.loc[cond_1, "Ni_4C_min"] = df.loc[cond_1, "Ni_4.1C_min"]
    df.loc[cond_1, "Ni_4C_max"] = df.loc[cond_1, "Ni_4.1C_max"]
    df.loc[cond_1, "Ni_4C"] = df.loc[cond_1, "Ni_4.1C"]

    df.loc[cond_2, "Ni_4C_min"] = df.loc[cond_2, "Ni_4.2C_min"]
    df.loc[cond_2, "Ni_4C_max"] = df.loc[cond_2, "Ni_4.2C_max"]
    df.loc[cond_2, "Ni_4C"] = df.loc[cond_2, "Ni_4.2C"]

    df.loc[cond_3, "Ni_4T_min"] = df.loc[cond_3, "Ni_4.2T_min"]
    df.loc[cond_3, "Ni_4T_max"] = df.loc[cond_3, "Ni_4.2T_max"]
    df.loc[cond_3, "Ni_4T"] = df.loc[cond_3, "Ni_4.2T"]

    df.loc[cond_4, "Ni_4T_min"] = df.loc[cond_4, "Ni_4.1T_min"]
    df.loc[cond_4, "Ni_4T_max"] = df.loc[cond_4, "Ni_4.1T_max"]
    df.loc[cond_4, "Ni_4T"] = df.loc[cond_4, "Ni_4.1T"]

    df.loc[cond_5, "Ni_5C_min"] = df.loc[cond_5, "Ni_5.1C_min"]
    df.loc[cond_5, "Ni_5C_max"] = df.loc[cond_5, "Ni_5.1C_max"]
    df.loc[cond_5, "Ni_5C"] = df.loc[cond_5, "Ni_5.1C"]

    df.loc[cond_6, "Ni_5C_min"] = df.loc[cond_6, "Ni_5.2C_min"]
    df.loc[cond_6, "Ni_5C_max"] = df.loc[cond_6, "Ni_5.2C_max"]
    df.loc[cond_6, "Ni_5C"] = df.loc[cond_6, "Ni_5.2C"]

    df.loc[cond_7, "Ni_5T_min"] = df.loc[cond_7, "Ni_5.2T_min"]
    df.loc[cond_7, "Ni_5T_max"] = df.loc[cond_7, "Ni_5.2T_max"]
    df.loc[cond_7, "Ni_5T"] = df.loc[cond_7, "Ni_5.2T"]

    df.loc[cond_8, "Ni_5T_min"] = df.loc[cond_8, "Ni_5.1T_min"]
    df.loc[cond_8, "Ni_5T_max"] = df.loc[cond_8, "Ni_5.1T_max"]
    df.loc[cond_8, "Ni_5T"] = df.loc[cond_8, "Ni_5.1T"]

    df.loc[cond_9, "Ni_6C_min"] = df.loc[cond_9, "Ni_6.1C_min"]
    df.loc[cond_9, "Ni_6C_max"] = df.loc[cond_9, "Ni_6.1C_max"]
    df.loc[cond_9, "Ni_6C"] = df.loc[cond_9, "Ni_6.1C"]

    df.loc[cond_10, "Ni_6C_min"] = df.loc[cond_10, "Ni_6.2C_min"]
    df.loc[cond_10, "Ni_6C_max"] = df.loc[cond_10, "Ni_6.2C_max"]
    df.loc[cond_10, "Ni_6C"] = df.loc[cond_10, "Ni_6.2C"]

    df.loc[cond_11, "Ni_6T_min"] = df.loc[cond_11, "Ni_6.2T_min"]
    df.loc[cond_11, "Ni_6T_max"] = df.loc[cond_11, "Ni_6.2T_max"]
    df.loc[cond_11, "Ni_6T"] = df.loc[cond_11, "Ni_6.2T"]

    df.loc[cond_12, "Ni_6T_min"] = df.loc[cond_12, "Ni_6.1T_min"]
    df.loc[cond_12, "Ni_6T_max"] = df.loc[cond_12, "Ni_6.1T_max"]
    df.loc[cond_12, "Ni_6T"] = df.loc[cond_12, "Ni_6.1T"]

    return df
