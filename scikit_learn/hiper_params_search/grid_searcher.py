import time

from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

from scikit_learn.hiper_params_search.common_searcher import CommonHiperParamsSearcher, Searcher


class GridCVHipperParamsSearcher(CommonHiperParamsSearcher):

    def __init__(self, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)

    def search_hiper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> Searcher:
        self.start_search_parameter_time = time.time()

        search = GridSearchCV(estimator=estimator,
                              param_grid=params,
                              cv=cv,
                              n_jobs=self.n_jobs,
                              verbose=self.log_level,
                              scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search

class HalvingGridCVHipperParamsSearcher(CommonHiperParamsSearcher):

    def __init__(self,
                 resource='n_samples',
                 max_resources='auto',
                 min_resources='exhaust',
                 n_jobs: int = -1,
                 log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources

    def search_hiper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> Searcher:
        self.start_search_parameter_time = time.time()

        search = HalvingGridSearchCV(estimator=estimator,
                                     param_grid=params,
                                     cv=cv,
                                     n_jobs=self.n_jobs,
                                     verbose=self.log_level,
                                     scoring=scoring,
                                     resource=self.resource,
                                     max_resources=self.max_resources,
                                     min_resources=self.min_resources)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search