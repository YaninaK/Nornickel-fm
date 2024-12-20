import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_data"]

PATH = ""
FOLDER_PATH = "data/01_raw/"
FILE_NAME = "df_hack_final.csv"


def load_data(
    path: Optional[str] = None,
    folder_path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:

    if path is None:
        path = PATH
    if folder_path is None:
        folder_path = FOLDER_PATH
    if file_name is None:
        file_name = f"{path}{folder_path}{FILE_NAME}"

    df = pd.read_csv(file_name)

    return df
