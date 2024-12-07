import logging
from typing import Optional

import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["define_process_model_fm1"]

from . import MODEL_CONFIG


def get_model_fm_1(
    X_fm1: pd.DataFrame, config: Optional[dict] = None
) -> tf.keras.Model:
    """
    Генерирует модель для выявления закономерностей процесса флотации
    первой флотомашины.

    """
    if config is None:
        config = MODEL_CONFIG

    norm_Cu_oreth = tf.keras.layers.Normalization(axis=None, name="norm_Cu_oreth")
    norm_Cu_oreth.adapt(X_fm1["Cu_oreth"].infer_objects(copy=False).fillna(0).values)

    norm_Ni_oreth = tf.keras.layers.Normalization(axis=None, name="norm_Ni_oreth")
    norm_Ni_oreth.adapt(X_fm1["Ni_oreth"].infer_objects(copy=False).fillna(0).values)

    norm_Ore_mass = tf.keras.layers.Normalization(axis=None, name="norm_Ore_mass")
    norm_Ore_mass.adapt(X_fm1["Ore_mass"].infer_objects(copy=False).fillna(0).values)

    norm_Mass_1 = tf.keras.layers.Normalization(axis=None, name="norm_Mass_1")
    norm_Mass_1.adapt(X_fm1["Mass_1"].infer_objects(copy=False).fillna(0).values)

    norm_Dens_1 = tf.keras.layers.Normalization(axis=None, name="norm_Dens_1")
    norm_Dens_1.adapt(X_fm1["Dens_1"].infer_objects(copy=False).fillna(0).values)

    norm_Cu_1C = tf.keras.layers.Normalization(axis=None, name="norm_Cu_1C")
    norm_Cu_1C.adapt(X_fm1["Cu_1C"].infer_objects(copy=False).fillna(0).values)

    norm_Ni_1C = tf.keras.layers.Normalization(axis=None, name="norm_Ni_1C")
    norm_Ni_1C.adapt(X_fm1["Ni_1C"].infer_objects(copy=False).fillna(0).values)

    inputs = {
        "Cu_oreth": tf.keras.Input(shape=(1,), dtype=int, name=f"Cu_oreth"),
        "Ni_oreth": tf.keras.Input(shape=(1,), dtype=float, name=f"Ni_oreth"),
        "Ore_mass": tf.keras.Input(shape=(1,), dtype=int, name=f"Ore_mass"),
        "Mass_1": tf.keras.Input(shape=(1,), dtype=float, name=f"Mass_1"),
        "Dens_1": tf.keras.Input(shape=(1,), dtype=int, name=f"Dens_1"),
        "Cu_1C": tf.keras.Input(shape=(1,), dtype=float, name=f"Cu_1C"),
        "Ni_1C": tf.keras.Input(shape=(1,), dtype=int, name=f"Ni_1C"),
    }

    layers = []
    layers.append(norm_Cu_oreth(inputs["Cu_oreth"]))
    layers.append(norm_Ni_oreth(inputs["Ni_oreth"]))
    layers.append(norm_Ore_mass(inputs["Ore_mass"]))
    layers.append(norm_Mass_1(inputs["Mass_1"]))
    layers.append(norm_Dens_1(inputs["Dens_1"]))
    layers.append(norm_Cu_1C(inputs["Cu_1C"]))
    layers.append(norm_Ni_1C(inputs["Ni_1C"]))

    all_features = tf.keras.layers.Concatenate(axis=-1, name="all_features")(layers)

    X = tf.keras.layers.Dense(
        config["fm_1_units_max"], activation="relu", name="dense_1"
    )(all_features)
    X = tf.keras.layers.Dense(
        config["fm_1_units_min"], activation="relu", name="dense_2"
    )(X)
    outputs = tf.keras.layers.Dense(
        config["fm_1_n_output_units"], activation="relu", name="output"
    )(X)
    model = tf.keras.Model(inputs, outputs)

    return model
