
from abc import abstractmethod, ABC
from typing import TypeVar

from sklearn.model_selection._search import BaseSearchCV

Searcher = TypeVar('Searcher', bound=BaseSearchCV)

class CommonHiperParamsSearcher(ABC):
    """
    Classe wrapper utilizada pelas implementações específicas de busca de parâmetros.
    """

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param n_jobs: Número de threads utilizado no processo de busca das features, se informado -1 serão utilizadas
        todas as threads. Não são todas as implementações que utilizam esse valor no processamento.

        :param log_level: Define quanto log será exibido durante o processamento, os valores vão de 1 até 3.

        Atributos Internos:
            start_search_parameter_time (int): Armazena o tempo de início da busca de parâmetros, em segundos.

            end_search_parameter_time (int): Armazena o tempo de término da busca de parâmetros, em segundos.
        """

        self.n_jobs = n_jobs
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    @abstractmethod
    def search_hiper_parameters(self,
                                estimator,
                                params,
                                data_x,
                                data_y,
                                cv,
                                scoring: str) -> Searcher:
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