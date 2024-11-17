import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

from wrappers.scikit_learn.hiper_params_search.common_hyper_params_searcher import ScikitLearnCommonHyperParamsSearcher, \
    ScikitLearnSearcher


class ScikitLearnGridCVHyperParamsSearcher(ScikitLearnCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca GridSearchCV o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
    """

    def __init__(self, n_jobs: int = -1, log_level: int = 0):
        super().__init__(n_jobs, log_level)

    def search_hyper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> ScikitLearnSearcher:
        self.start_search_parameter_time = time.time()

        search = GridSearchCV(estimator=estimator,
                              param_grid=params,
                              cv=cv,
                              n_jobs=self.n_jobs,
                              verbose=self.log_level,
                              scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search

class ScikitLearnHalvingGridCVHyperParamsSearcher(ScikitLearnCommonHyperParamsSearcher):
    """
    Implementação wrapper da busca GridSearchCV o qual é detalhado na `documentação do scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html>`_.
    """

    def __init__(self,
                 resource='n_samples',
                 max_resources='auto',
                 min_resources='exhaust',
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param resource: Especifica o recurso a ser usado para avaliar o desempenho de cada combinação de hiperparâmetros.
        Pode ser um número de amostras, tempo de treinamento ou qualquer outro recurso que se deseja limitar durante a
        validação.

        :param max_resources: Define a quantidade máxima do recurso que pode ser utilizado durante a execução. Se você
        especificar max_resources=100, a busca usará no máximo 100 amostras (ou outra métrica definida em resource)
        durante a validação de cada combinação.

        :param min_resources: Determina a quantidade mínima do recurso que deve ser usada na primeira iteração da busca.
        É o ponto de partida que garante que a busca comece com uma base sólida de amostras ou recursos.
        """

        super().__init__(n_jobs, log_level)
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources

    def search_hyper_parameters(self, estimator, params, data_x, data_y, cv, scoring: str) -> ScikitLearnSearcher:
        self.start_search_parameter_time = time.time()

        search = HalvingGridSearchCV(estimator=estimator,
                                     param_grid=params,
                                     cv=cv,
                                     n_jobs=self.n_jobs,
                                     verbose=self.log_level,
                                     scoring=scoring,
                                     resource=self.resource,
                                     max_resources=self.max_resources,
                                     min_resources=self.min_resources)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search