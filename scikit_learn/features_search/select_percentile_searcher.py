import time

from sklearn.feature_selection import SelectPercentile

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class SelectPercentileSearcher(CommonFeaturesSearcher):

    def __init__(self, percent: int, score_func, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.percent = percent
        self.score_func = score_func

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = SelectPercentile(score_func=self.score_func, percentile=self.percent)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]