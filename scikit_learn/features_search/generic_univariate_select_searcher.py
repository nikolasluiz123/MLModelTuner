import time

from sklearn.feature_selection import GenericUnivariateSelect

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class GenericUnivariateSelectSearcher(CommonFeaturesSearcher):

    def __init__(self, score_func, mode, features_number: int = 1e-5, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.score_func = score_func
        self.mode = mode
        self.features_number = features_number

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = GenericUnivariateSelect(score_func=self.score_func, mode=self.mode, param=self.features_number)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]