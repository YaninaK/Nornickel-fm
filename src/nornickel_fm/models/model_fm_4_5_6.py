import logging
from typing import Optional

import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["define_process_model_fm_4_5_6"]

from . import MODEL_CONFIG


def get_model_fm_4_5_6(
    X_fm_4_5_6: pd.DataFrame, config: Optional[dict] = None
) -> tf.keras.Model:
    """
    Генерирует модель для выявления закономерностей процесса флотации
    4, 5 и 6 флотомашин.

    """
    if config is None:
        config = MODEL_CONFIG

    norm_Mass = tf.keras.layers.Normalization(axis=None, name="norm_Mass")
    norm_Mass.adapt(X_fm_4_5_6["Mass"].infer_objects(copy=False).fillna(0).values)

    norm_Vol = tf.keras.layers.Normalization(axis=None, name="norm_Vol")
    norm_Vol.adapt(X_fm_4_5_6["Vol"].infer_objects(copy=False).fillna(0).values)

    norm_Dens = tf.keras.layers.Normalization(axis=None, name="norm_Dens")
    norm_Dens.adapt(X_fm_4_5_6["Dens"].infer_objects(copy=False).fillna(0).values)

    norm_Ni_F = tf.keras.layers.Normalization(axis=None, name="norm_Ni_F")
    norm_Ni_F.adapt(X_fm_4_5_6["Ni_F"].infer_objects(copy=False).fillna(0).values)

    norm_Ni_C = tf.keras.layers.Normalization(axis=None, name="norm_Ni_C")
    norm_Ni_C.adapt(X_fm_4_5_6["Ni_C"].infer_objects(copy=False).fillna(0).values)

    norm_Ni_T = tf.keras.layers.Normalization(axis=None, name="norm_Ni_T")
    norm_Ni_T.adapt(X_fm_4_5_6["Ni_T"].infer_objects(copy=False).fillna(0).values)

    norm_Cu_F = tf.keras.layers.Normalization(axis=None, name="norm_Cu_F")
    norm_Cu_F.adapt(X_fm_4_5_6["Cu_F"].infer_objects(copy=False).fillna(0).values)

    intlookup_fm = tf.keras.layers.IntegerLookup(
        output_mode="one_hot", name=f"intlookup_fm"
    )
    intlookup_fm.adapt(X_fm_4_5_6["fm"])

    inputs = {
        "Mass": tf.keras.Input(shape=(1,), dtype=int, name=f"Mass"),
        "Vol": tf.keras.Input(shape=(1,), dtype=int, name=f"Vol"),
        "Dens": tf.keras.Input(shape=(1,), dtype=float, name=f"Dens"),
        "Ni_F": tf.keras.Input(shape=(1,), dtype=float, name=f"Ni_F"),
        "Ni_C": tf.keras.Input(shape=(1,), dtype=float, name=f"Ni_C"),
        "Ni_T": tf.keras.Input(shape=(1,), dtype=int, name=f"Ni_T"),
        "Cu_F": tf.keras.Input(shape=(1,), dtype=int, name=f"Cu_F"),
        "fm": tf.keras.Input(shape=(1,), dtype=int, name=f"fm"),
    }

    layers = []
    layers.append(norm_Mass(inputs["Mass"]))
    layers.append(norm_Vol(inputs["Vol"]))
    layers.append(norm_Dens(inputs["Dens"]))
    layers.append(norm_Ni_F(inputs["Ni_F"]))
    layers.append(norm_Ni_C(inputs["Ni_C"]))
    layers.append(norm_Ni_T(inputs["Ni_T"]))
    layers.append(norm_Cu_F(inputs["Cu_F"]))
    layers.append(intlookup_fm(inputs["fm"]))

    all_features = tf.keras.layers.Concatenate(axis=-1, name="all_features")(layers)

    X = tf.keras.layers.Dense(
        config["fm_4_5_6_units_max"], activation="relu", name="dense_1"
    )(all_features)
    X = tf.keras.layers.Dense(
        config["fm_4_5_6_units_min"], activation="relu", name="dense_2"
    )(X)
    outputs = tf.keras.layers.Dense(
        config["fm_4_5_6_n_output_units"], activation="relu", name="output"
    )(X)
    model = tf.keras.Model(inputs, outputs)

    return model
