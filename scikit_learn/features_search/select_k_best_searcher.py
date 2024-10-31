import time

from sklearn.feature_selection import SelectKBest

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class SelectKBestSearcher(CommonFeaturesSearcher):

    def __init__(self, features_number: int, score_func, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.features_number = features_number
        self.score_func = score_func

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = SelectKBest(score_func=self.score_func, k=self.features_number)
        searcher.fit_transform(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]
