from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ProcessedDta:
    processed_x: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: np.ndarray = field(default_factory=np.array)
    categorical_feature_cols: np.ndarray = field(default_factory=np.array)
    categorical_feature_mapping: List[Dict[Any, Any]] = field(default_factory=list)
    feature_weights: Optional[str] = None
    feature_weights_variables: Optional[str] = None
    deep_feature_weights: Optional[str] = None
    deep_feature_weights_variables: Optional[str] = None
    observation_weights: Optional[str] = None
    symmetric: Optional[str] = None
    monotonic_constraints: Optional[str] = None
    linear_feature_cols: np.ndarray = field(default_factory=np.array)
    groups_mapping: Optional[Dict[str, Any]] = None
    groups: Optional[str] = None
    col_means: np.ndarray = field(default_factory=np.array)
    col_sd: np.ndarray = field(default_factory=np.array)
    has_nas: bool = False
    n_observations: int = 0
    num_columns: int = 0
    feat_names: Optional[np.ndarray] = None
