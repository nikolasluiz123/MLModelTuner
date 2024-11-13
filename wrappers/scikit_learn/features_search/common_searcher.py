from abc import ABC, abstractmethod


class CommonFeaturesSearcher(ABC):
    """
    Classe utilizada nas implementações wrapper das implementações de busca das melhores features do scikit-learn.
    """

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param n_jobs: Número de threads utilizado no processo de busca das features, se informado -1 serão utilizadas
        todas as threads. Não são todas as implementações que utilizam esse valor no processamento.

        :param log_level: Define quanto log será exibido durante o processamento, os valores vão de 1 até 3.

        Atributos Internos:
            start_search_features_time (int): Armazena o tempo de início da busca de features, em segundos.

            end_search_features_time (int): Armazena o tempo de término da busca de features, em segundos.
        """

        self.n_jobs = n_jobs
        self.log_level = log_level

        self.start_search_features_time = 0
        self.end_search_features_time = 0

    @abstractmethod
    def select_features(self, data_x, data_y, scoring=None, estimator=None, cv=None):
        """
        Função que pode ser utilizada para realizar a selação das features. Normalmente o retorno será `data_x` selecionando
        apenas as colunas que representam as features retornadas pela implementação de busca.

        :param data_x: Dados declarados como features, de onde serão selecionados as melhores.

        :param data_y: Dados declarados como target.

        :param scoring: Forma de avaliação dos resultados quando a implementação suporta validação cruzada. Opcional, pois
        nem todas as implementação utilizam a validação cruzada.

        :param estimator: Modelo que algumas implementações de busca de features utilizam para melhorar a acertividade
        da seleção das melhores features

        :param cv: Definição da validação cruzada, por exemplo, KFold ou StratifiedKFold. Opcional, pois
        nem todas as implementações utilizam validação cruzada.
        """