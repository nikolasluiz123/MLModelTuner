import time

from sklearn.feature_selection import RFECV, RFE

from wrappers.scikit_learn import CommonFeaturesSearcher


class RecursiveFeatureSearcher(CommonFeaturesSearcher):
    """
    Implementação wrapper do algoritmo RFE o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>`_.
    """

    def __init__(self,
                 features_number: int = None,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param features_number: Número exato de features que serão buscadas. Se passar None a metade das features será
        retornada.
        """
        super().__init__(n_jobs, log_level)

        self.features_number = features_number

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
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
    """
    Implementação wrapper do algoritmo RFECV o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html>`_.
    """

    def __init__(self,
                 min_features: int = 3,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param min_features: Número mínimo de features esperado.
        """
        super().__init__(n_jobs, log_level)

        self.min_feeatures = min_features

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        if estimator is None:
            raise Exception("The parameter estimator can't be None")

        if cv is None:
            raise Exception("The parameter cv can't be None")

        if scoring is None:
            raise Exception("The parameter scoring can't be None")

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
