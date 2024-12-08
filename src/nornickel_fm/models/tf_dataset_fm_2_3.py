import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["generate_tf_dataset_for_model_fm_2_3"]

from features import CONFIG


def get_tf_dataset_fm_2_3(df: pd.DataFrame) -> tf.data.Dataset:
    """
    Создает tensorflow dataset для fm_2_3 на основе генератора данных.
    """
    ds = tf.data.Dataset.from_generator(
        lambda: generator_fm_2_3(df),
        output_signature=(
            {
                "Mass": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Mass"),
                "Dens": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Dens"),
                "Cu_F": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Cu_F"),
                "Ni_F": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_F"),
                "Cu_C": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Cu_C"),
                "Ni_C": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_C"),
                "Cu_T": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Cu_T"),
                "Ni_T": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_T"),
                "fm": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="fm"),
            },
            tf.TensorSpec(
                shape=(2,),
                dtype=tf.float64,
                name="output",
            ),
        ),
    )
    return ds


def generator_fm_2_3(df: pd.DataFrame, config: Optional[dict] = None):
    """
    Генерирует данные для обучения модели fm_2_3.
    """
    if config is None:
        config = CONFIG

    for i in df.index:
        inp = df.loc[i, config["fm_2_3_feature_cols"]].values
        inp = np.expand_dims(inp, axis=1)
        inputs = {
            "Mass": inp[0].astype(float),
            "Dens": inp[1].astype(float),
            "Cu_F": inp[2].astype(float),
            "Ni_F": inp[3].astype(float),
            "Cu_C": inp[4].astype(float),
            "Ni_C": inp[5].astype(float),
            "Cu_T": inp[6].astype(float),
            "Ni_T": inp[7].astype(float),
            "fm": inp[8].astype(int),
        }
        label = np.array(df.loc[i, config["fm_2_3_target_cols"]])
        yield inputs, label
