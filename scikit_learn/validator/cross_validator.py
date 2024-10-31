import time

import numpy as np
from sklearn.model_selection import cross_val_score

from scikit_learn.validator.common_validator import BaseValidator, Result
from scikit_learn.validator.results.cross_validation import CrossValidationResult


class CrossValidator(BaseValidator):

    def __init__(self,
                 log_level: int = 0,
                 n_jobs: int = -1):
        super().__init__(log_level, n_jobs)

    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> Result | None:
        if cv is None:
            raise Exception("The parameter cv can't be None")

        if scoring is None:
            raise Exception("The parameter scoring can't be None")

        self.start_best_model_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=cv,
                                 n_jobs=self.n_jobs,
                                 verbose=self.log_level,
                                 scoring=scoring)

        self.end_best_model_validation = time.time()

        result = CrossValidationResult(
            mean=np.mean(scores),
            standard_deviation=np.std(scores),
            median=np.median(scores),
            variance=np.var(scores),
            standard_error=np.std(scores) / np.sqrt(len(scores)),
            min_max_score=(round(float(np.min(scores)), 4), round(float(np.max(scores)), 4)),
            estimator=searcher.best_estimator_,
            scoring=scoring
        )

        return result
