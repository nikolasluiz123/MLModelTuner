import time

from sklearn.feature_selection import SelectKBest

from wrappers.scikit_learn.features_search.common_feature_searcher import ScikitLearnCommonFeaturesSearcher


class ScikitLearnSelectKBestSearcher(ScikitLearnCommonFeaturesSearcher):
    """
    Implementação wrapper do algoritmo SelectKBest o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>`_.
    """

    def __init__(self, features_number: int, score_func, n_jobs: int = -1, log_level: int = 0):
        """
        :param features_number: Número exato de features desejadas.

        :param score_func: Função que será utilizada para a escolha das features. Uma opção de uso para classificação é
        f_classif, enquanto para regressão poderia ser utilizado f_regression.
        """
        super().__init__(n_jobs, log_level)
        self.features_number = features_number
        self.score_func = score_func

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = SelectKBest(score_func=self.score_func, k=self.features_number)
        searcher.fit_transform(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]
