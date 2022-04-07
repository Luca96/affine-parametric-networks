import numpy as np


def retrieve_stat(stats: dict, which: str, columns: list) -> np.ndarray:
    return np.array([stats[col][which] for col in columns])


def retrieve_clip(ranges: dict,  columns: list) -> np.ndarray:
    return np.array([ranges[col] for col in columns])
