import time
from abc import ABC, abstractmethod

from keras.src.callbacks import Callback
from keras_tuner import HyperModel

from wrappers.common.hyper_params_searcher.common_hyper_params_search import CommonHyperParamsSearch


class KerasCommonHyperParamsSearcher(CommonHyperParamsSearch):
    """
    Classe base para implementação das diferentes buscas de hiperparâmetros utilizando a biblioteca keras.
    """

    def __init__(self,
                 objective: str | list[str],
                 directory: str,
                 project_name: str,
                 epochs: int,
                 batch_size: int,
                 callbacks: list[Callback],
                 log_level: int = 0):
        """
        :param objective: O que deseja maximizar ou minimizar durante a busca dos melhores parâmetros do modelo. Pode ser
                          um único valor ou uma lista, por exemplo, accuracy.

        :param directory: Diretório que será utilizado internamento pelas implementações do keras para salvar os projetos.

        :param project_name: Nome do projeto, utilizado para criar um sub diretório abaixo de `directory` que vai conter
                             as tentativas de busca de hiperparâmetros de um modelo. Normalmente é definido um `project_name`
                             para cada execução de modelo, dessa forma as tentativas ficam separadas e o keras irá recomeçar
                             a execução caso a operação seja parada sem ser finalizada por algum motivo.

        :param epochs: Quantidade de épocas que serão executadas em cada uma das tentativas.

        :param batch_size: Quantidade de dados que serão enviados de uma só vez para que sejam processados pela busca.
                           O valor definido aqui depende do hardware onde estiver executando e de que tipo de dado
                           estiver trabalhando, por sugestão, comece com valores entre 16 e 64 e verifique os logs para
                           garantir que não haverão problemas de memória, se tudo correr bem é interessante ir aumentando
                           esse valor para descobrir os limites do seu hardware. Esse parâmetro é um dos que impacta no
                           desempenho da busca.

        :param callbacks: Lista de callbacks utilizados internamente nas implementações de busca de hiperparâmetros do
                          keras. Existem diversas possíbilidades, por exemplo, EarlyStopping que faz o processamento
                          das épocas daquela tentativa ser interrompido caso o modelo não esteja melhorando.
        """
        super().__init__(log_level)

        self.objective = objective
        self.directory = directory
        self.project_name = project_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

    def process(self,
                train_data,
                validation_data,
                model: HyperModel):
        """
        Função utilizada para realizar a busca dos hiperparâmetros do modelo.

        :param train_data: Conjunto de dados de treino
        :param validation_data: Conjunto de dados de validação
        :param model: Implementação de HyperModel do keras que contem a estrutura da rede neural que será avaliada
        """

        self.start_search_parameter_time = time.time()

        model = self._on_execute(train_data=train_data,
                                 validation_data=validation_data,
                                 model=model)

        self.end_search_parameter_time = time.time()

        return model

    @abstractmethod
    def _on_execute(self,
                    train_data,
                    validation_data,
                    model: HyperModel):
        """
        Função interna para realizar os processos específicos de cada implementação de busca de hiperparâmetros utilizando
        a biblioteca keras.

        :param train_data: Conjunto de dados de treino
        :param validation_data: Conjunto de dados de validação
        :param model: Implementação de HyperModel do keras que contem a estrutura da rede neural que será avaliada
        """

    @abstractmethod
    def get_fields_oracle_json_file(self) -> list[str]:
        """
        Cada implementação de busca de hiperparâmetros possui suas peculiaridades, dessa forma, os dados que serão salvos
        no oracle.json (arquivo mantido internamente pelo keras) variam um pouco.

        Essa função é responsável por retornar uma lista com os campos específicos da implementação de busca, para que
        possam ser recuperados os valores e salvos no histórico gerenciado pela implementação de CommonHistoryManager.
        """
