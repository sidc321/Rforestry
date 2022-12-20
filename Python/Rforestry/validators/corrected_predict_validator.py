from typing import Final

import numpy as np

from .base_validator import BaseValidator


class CorrectedPredictValidator(BaseValidator):

    DEFAULT_NROUNDS: Final[int] = 0
    DEFAULT_LINEAR: Final[bool] = True
    DEFAULT_FEATS: Final = None

    def validate_nrounds(self, **kwargs):
        nrounds: int = kwargs.get("nrounds", __class__.DEFAULT_NROUNDS)
        linear: bool = kwargs.get("linear", __class__.DEFAULT_LINEAR)

        # Check allowed settings for the bias correction
        if nrounds < 1 and not linear:
            raise ValueError(
                "We must do at least one round of bias corrections, with either linear = True or nrounds > 0."
            )

        if nrounds < 0 or not isinstance(nrounds, int):
            raise ValueError("nrounds must be a non negative integer.")

    def validate_feats(self, *args, **kwargs):
        _self = args[0]

        feats: bool = kwargs.get("feats", __class__.DEFAULT_FEATS)

        if feats is not None:
            if any(
                not isinstance(x, (int, np.integer))
                or x < -_self.processed_dta.num_columns  # pylint: disable=invalid-unary-operand-type
                or x >= _self.processed_dta.num_columns
                for x in feats
            ):
                raise ValueError("feats must be  a integer between -ncol and ncol(x)-1")

    def __call__(self, *args, **kwargs):

        self.validate_nrounds(*kwargs)
        self.validate_feats(*args, **kwargs)

        return self.function(self, *args, **kwargs)
