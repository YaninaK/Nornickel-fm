import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["generate_tf_dataset_for_model_fm_1"]

from features import CONFIG


def get_tf_dataset_fm_1(df: pd.DataFrame) -> tf.data.Dataset:
    """
    Создает tensorflow dataset для fm_1 на основе генератора данных.
    """
    ds = tf.data.Dataset.from_generator(
        lambda: generator_fm_1(df),
        output_signature=(
            {
                "Cu_oreth": tf.TensorSpec(
                    shape=(1,), dtype=tf.float64, name="Cu_oreth"
                ),
                "Ni_oreth": tf.TensorSpec(
                    shape=(1,), dtype=tf.float64, name="Ni_oreth"
                ),
                "Ore_mass": tf.TensorSpec(
                    shape=(1,), dtype=tf.float64, name="Ore_mass"
                ),
                "Mass_1": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Mass_1"),
                "Dens_1": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Dens_1"),
                "Cu_1C": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Cu_1C"),
                "Ni_1C": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="Ni_1C"),
            },
            tf.TensorSpec(
                shape=(4,),
                dtype=tf.float64,
                name="output",
            ),
        ),
    )
    return ds


def generator_fm_1(df: pd.DataFrame, config: Optional[dict] = None):
    """
    Генерирует данные для обучения модели fm_1.
    """
    if config is None:
        config = CONFIG

    for i in df.index:
        inp = df.loc[i, config["fm_1_feature_cols"]].values
        inp = np.expand_dims(inp, axis=1)
        inputs = {
            "Cu_oreth": inp[0].astype(float),
            "Ni_oreth": inp[1].astype(float),
            "Ore_mass": inp[2].astype(float),
            "Mass_1": inp[3].astype(float),
            "Dens_1": inp[4].astype(float),
            "Cu_1C": inp[5].astype(float),
            "Ni_1C": inp[6].astype(float),
        }
        label = np.array(df.loc[i, config["fm_1_target_cols"]])
        yield inputs, label
