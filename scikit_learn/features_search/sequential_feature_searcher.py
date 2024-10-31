import time

from sklearn.feature_selection import SequentialFeatureSelector

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class SequentialFeatureSearcher(CommonFeaturesSearcher):

    def __init__(self, number_features='auto', n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)
        self.number_features = number_features

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        if estimator is None:
            raise Exception("The parameter estimator can't be None")

        if cv is None:
            raise Exception("The parameter cv can't be None")

        self.start_search_features_time = time.time()

        searcher = SequentialFeatureSelector(estimator=estimator,
                                             cv=cv,
                                             scoring=scoring,
                                             n_jobs=self.n_jobs,
                                             n_features_to_select=self.number_features)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]