
from abc import abstractmethod, ABC
from typing import TypeVar

from sklearn.model_selection._search import BaseSearchCV

Searcher = TypeVar('Searcher', bound=BaseSearchCV)

class CommonHiperParamsSearcher(ABC):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0):
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    @abstractmethod
    def search_hiper_parameters(self,
                                estimator,
                                params,
                                data_x,
                                data_y,
                                cv,
                                scoring: str) -> Searcher:
        ...