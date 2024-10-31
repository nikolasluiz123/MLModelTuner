import time

from sklearn.feature_selection import RFECV, RFE

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class RecursiveFeatureSearcher(CommonFeaturesSearcher):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0,
                 features_number: int = None):
        super().__init__(n_jobs, log_level)

        self.features_number = features_number

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        if estimator is None:
            raise Exception("The parameter estimator can't be None")

        self.start_search_features_time = time.time()

        searcher = RFE(estimator=estimator,
                       verbose=self.log_level,
                       n_features_to_select=self.features_number)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]


class RecursiveFeatureCVSearcher(CommonFeaturesSearcher):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0,
                 min_features: int = 3):
        super().__init__(n_jobs, log_level)

        self.min_feeatures = min_features

    def select_features(self, data_x, data_y, scoring, estimator=None, cv=None):
        if estimator is None:
            raise Exception("The parameter estimator can't be None")

        if cv is None:
            raise Exception("The parameter cv can't be None")

        self.start_search_features_time = time.time()

        searcher = RFECV(estimator=estimator,
                         cv=cv,
                         scoring=scoring,
                         n_jobs=self.n_jobs,
                         verbose=self.log_level,
                         min_features_to_select=self.min_feeatures)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]
