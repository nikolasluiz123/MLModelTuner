import time

from sklearn.feature_selection import SelectPercentile

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class SelectPercentileSearcher(CommonFeaturesSearcher):
    """
    Implementação wrapper do algoritmo SelectPercentile o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html>`_.
    """

    def __init__(self, percent: int, score_func, n_jobs: int = -1, log_level: int = 0):
        """
        :param percent: Percentual de features a serem selecionadas.

        :param score_func: Função que será utilizada para a escolha das features. Uma opção de uso para classificação é
        f_classif, enquanto para regressão poderia ser utilizado f_regression.
        """

        super().__init__(n_jobs, log_level)
        self.percent = percent
        self.score_func = score_func

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = SelectPercentile(score_func=self.score_func, percentile=self.percent)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]