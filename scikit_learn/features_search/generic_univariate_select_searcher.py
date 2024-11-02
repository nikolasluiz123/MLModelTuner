import time

from sklearn.feature_selection import GenericUnivariateSelect

from scikit_learn.features_search.common_searcher import CommonFeaturesSearcher


class GenericUnivariateSelectSearcher(CommonFeaturesSearcher):
    """
    Implementação wrapper do algoritmo GenericUnivariateSelect o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html>`_.
    """

    def __init__(self, score_func, mode, mode_param: int = 1e-5, n_jobs: int = -1, log_level: int = 0):
        """
        :param score_func: Função que será utilizada para a escolha das features. Uma opção de uso para classificação é f_classif,
        enquanto para regressão poderia ser utilizado f_regression.

        :param mode: Esse parâmetro define a maneira como as features serão escolhidas, impactando a quantidade de
        features retidas.

        :param mode_param: Reperesenta o valor utilizado de maneiras diferentes dependendo do que for definido no mode.
        """

        super().__init__(n_jobs, log_level)
        self.score_func = score_func
        self.mode = mode
        self.mode_param = mode_param

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        self.start_search_features_time = time.time()

        searcher = GenericUnivariateSelect(score_func=self.score_func, mode=self.mode, param=self.mode_param)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]