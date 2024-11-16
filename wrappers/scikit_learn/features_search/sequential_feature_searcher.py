import time

from sklearn.feature_selection import SequentialFeatureSelector

from wrappers.scikit_learn.features_search.common_feature_searcher import ScikitLearnCommonFeaturesSearcher


class ScikitLearnSequentialFeatureSearcher(ScikitLearnCommonFeaturesSearcher):
    """
    Implementação wrapper do algoritmo SequentialFeatureSelector o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html>`_.
    """

    def __init__(self,
                 number_features='auto',
                 tolerance=None,
                 direction='forward',
                 n_jobs: int = -1,
                 log_level: int = 0, ):
        """
        :param number_features: Pode ser um número exato de features desejado ou, se preferir, é possível passar `auto` e
        definir um valor para `tol`.

        :param tolerance: Quando `number_features` for definido como `auto` deve ser definido um valor para esse parâmetro,
        para que seja determinado um critério de parada da busca de features. O valor definido é o mínimo de melhoria
        esperado para que a seleção continue.

        :param direction: Define a direção do processo de seleção, `forward` ou `backward`. Se `forward` for escolhido,
        o processo inicia com 0 features e vai sendo incrementado, se `backward` for escolhido, o processo inicia com
        todas as features e elas vão sendo removidas.
        """

        super().__init__(n_jobs, log_level)
        self.number_features = number_features
        self.tolerance = tolerance
        self.direction = direction

    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        if estimator is None:
            raise Exception("The parameter estimator can't be None")

        if cv is None:
            raise Exception("The parameter cv can't be None")

        if scoring is None:
            raise Exception("The parameter scoring can't be None")

        self.start_search_features_time = time.time()

        searcher = SequentialFeatureSelector(estimator=estimator,
                                             cv=cv,
                                             scoring=scoring,
                                             n_jobs=self.n_jobs,
                                             n_features_to_select=self.number_features,
                                             tol=self.tolerance,
                                             direction=self.direction)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]