
from abc import abstractmethod
from typing import TypeVar

from sklearn.model_selection._search import BaseSearchCV

from wrappers.common.hyper_params_searcher.common_hyper_params_search import CommonHyperParamsSearch

ScikitLearnSearcher = TypeVar('ScikitLearnSearcher', bound=BaseSearchCV)

class ScikitLearnCommonHyperParamsSearcher(CommonHyperParamsSearch):
    """
    Classe wrapper utilizada pelas implementações específicas envolvendo scikit-learn de busca de parâmetros.
    """

    def __init__(self, n_jobs: int = -1, log_level: int = 0):
        """
        :param n_jobs: Número de threads utilizado no processo de busca das features, se informado -1 serão utilizadas
                       todas as threads. Não são todas as implementações que utilizam esse valor no processamento.
        """

        super().__init__(log_level)
        self.n_jobs = n_jobs

    @abstractmethod
    def search_hyper_parameters(self,
                                estimator,
                                params,
                                data_x,
                                data_y,
                                cv,
                                scoring: str) -> ScikitLearnSearcher:
        """
        Função que deve realizar a seleção dos melhores parâmetros para o `estimator` definido. O retorno da função
        deve ser obrigatoriamente alguma implementação filha de `BaseSearchCV` que é a classe mais alta na hierarquia
        de implementações de busca do scikit-learn.

        :param estimator: Implementação que deseja buscar os parâmetros.
        :param params: Parâmetros e os valores que deseja buscar.
        :param data_x: Dados declarados como features, de onde serão selecionados as melhores.
        :param data_y: Dados declarados como target.
        :param scoring: Forma de avaliação dos resultados quando a implementação.
        :param cv: Definição da validação cruzada, por exemplo, KFold ou StratifiedKFold.
        """