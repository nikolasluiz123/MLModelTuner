import time

import numpy as np
from sklearn.model_selection import cross_val_score

from wrappers.common.history_manager.common_history_manager import CommonValResult
from wrappers.scikit_learn.validator.common_validator import ScikitLearnCommonValidator
from wrappers.scikit_learn.validator.results.cross_validation_result import ScikitLearnCrossValidationResult


class ScikitLearnCrossValidator(ScikitLearnCommonValidator[ScikitLearnCrossValidationResult]):
    """
    Validador que utiliza validação cruzada para avaliar modelos.

    A implementação é toda baseada na função `cross_val_score` do scikit-learn, a qual internamente realiza os processamentos
    de acordo com os parâmetros passados e, no fim, retorna um conjunto de scores que representa o que foi passado no
    atributo scoring.
    """

    def __init__(self,
                 log_level: int = 0,
                 n_jobs: int = -1):
        super().__init__(log_level, n_jobs)

    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> CommonValResult:
        if cv is None:
            raise Exception("The parameter cv can't be None")

        if scoring is None:
            raise Exception("The parameter scoring can't be None")

        self.start_validation_best_model_time = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=cv,
                                 n_jobs=self.n_jobs,
                                 verbose=self.log_level,
                                 scoring=scoring)

        self.end_validation_best_model_time = time.time()

        result = ScikitLearnCrossValidationResult(
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
