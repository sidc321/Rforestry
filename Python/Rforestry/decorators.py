import functools

import numpy as np
import pandas as pd


class DefaultsSetters:
    @staticmethod
    def predict(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            forest = args[0]

            if "seed" not in kwargs:
                kwargs["seed"] = forest.seed
            else:
                if (not isinstance(kwargs["seed"], int)) or kwargs["seed"] < 0:
                    raise ValueError("seed must be a nonnegative integer.")
            if "nthread" not in kwargs:
                kwargs["nthread"] = forest.nthread

            func(*args, **kwargs)

        return wrapper

    @staticmethod
    def fit(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            forest = args[0]
            x = (pd.DataFrame(args[1])).copy()
            _, ncol = x.shape

            if "symmetric" not in kwargs:
                kwargs["symmetric"] = np.zeros(ncol, dtype=np.ulonglong)
            if "monotonic_constraints" not in kwargs:
                kwargs["monotonic_constraints"] = np.zeros(ncol, dtype=np.intc)
            if "seed" not in kwargs:
                kwargs["seed"] = forest.seed
            else:
                if (not isinstance(kwargs["seed"], int)) or kwargs["seed"] < 0:
                    raise ValueError("seed must be a nonnegative integer.")
            if "lin_feats" not in kwargs:
                kwargs["lin_feats"] = np.arange(ncol, dtype=np.ulonglong)

            func(*args, **kwargs)

        return wrapper

    @staticmethod
    def constructor(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if "double_bootstrap" not in kwargs:
                if "oob_honest" in kwargs:
                    kwargs["double_bootstrap"] = kwargs["oob_honest"]
                else:
                    kwargs["double_bootstrap"] = False
            func(*args, **kwargs)

        return wrapper
