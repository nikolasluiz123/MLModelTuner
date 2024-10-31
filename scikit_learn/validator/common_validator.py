from abc import ABC, abstractmethod
from typing import TypeVar

from scikit_learn.validator.results.common import ValidationResult

Result = TypeVar('Result', bound=ValidationResult)

class BaseValidator(ABC):

    def __init__(self,
                 log_level: int = 1,
                 n_jobs: int = -1):
        self.log_level = log_level
        self.n_jobs = n_jobs

        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 cv=None,
                 scoring=None) -> Result | None:
        ...