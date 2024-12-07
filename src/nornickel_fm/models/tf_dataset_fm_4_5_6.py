import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["generate_tf_dataset_for_model_fm_4_5_6"]

from features import CONFIG


def get_tf_dataset_fm_4_5_6(df: pd.DataFrame) -> tf.data.Dataset:
    """
    Создает tensorflow dataset для fm_4_5_6 на основе генератора данных.
    """
    ds = tf.data.Dataset.from_generator(
        lambda: generator_fm_4_5_6(df),
        output_signature=(
            {
                "Mass": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Mass"),
                "Vol": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Vol"),
                "Dens": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Dens"),
                "Ni_F": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_F"),
                "Ni_C": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_C"),
                "Ni_T": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_T"),
                "Cu_F": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Cu_F"),
                "fm": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="fm"),
            },
            tf.TensorSpec(
                shape=(4,),
                dtype=tf.float64,
                name="output",
            ),
        ),
    )
    return ds


def generator_fm_4_5_6(df: pd.DataFrame, config: Optional[dict] = None):
    """
    Генерирует данные для обучения модели fm_4_5_6.
    """
    if config is None:
        config = CONFIG

    for i in df.index:
        inp = df.loc[i, config["fm_4_5_6_feature_cols"]].values
        inp = np.expand_dims(inp, axis=1)
        inputs = {
            "Mass": inp[0].astype(float),
            "Vol": inp[1].astype(float),
            "Dens": inp[2].astype(float),
            "Ni_F": inp[3].astype(float),
            "Ni_C": inp[4].astype(float),
            "Ni_T": inp[5].astype(float),
            "Cu_F": inp[6].astype(float),
            "fm": inp[7].astype(int),
        }
        label = np.array(df.loc[i, config["fm_4_5_6_target_cols"]])
        yield inputs, label
