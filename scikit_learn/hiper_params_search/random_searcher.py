import time

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

from scikit_learn.hiper_params_search.common_searcher import CommonHiperParamsSearcher, Searcher


class RandomCVHipperParamsSearcher(CommonHiperParamsSearcher):

    def __init__(self, number_iterations: int, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.number_iterations = number_iterations

    def search_hiper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> Searcher:
        self.start_search_parameter_time = time.time()

        search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=params,
                                    n_iter=self.number_iterations,
                                    cv=cv,
                                    n_jobs=self.n_jobs,
                                    verbose=self.log_level,
                                    scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search


class HalvingRandomCVHipperParamsSearcher(CommonHiperParamsSearcher):

    def __init__(self,
                 number_candidates: int,
                 resource='n_samples',
                 max_resources='auto',
                 min_resources='exhaust',
                 n_jobs: int = -1,
                 log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.number_candidates = number_candidates
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources

    def search_hiper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> Searcher:
        self.start_search_parameter_time = time.time()

        search = HalvingRandomSearchCV(estimator=estimator,
                                       param_distributions=params,
                                       cv=cv,
                                       n_jobs=self.n_jobs,
                                       verbose=self.log_level,
                                       scoring=scoring,
                                       n_candidates=self.number_candidates,
                                       resource=self.resource,
                                       max_resources=self.max_resources,
                                       min_resources=self.min_resources)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search
