import time
from abc import ABC, abstractmethod


class CommonDataPreProcessor(ABC):
    """
        Implementação para centralizar todos os processos que sejam necessários para a obtenção dos dados que serão
        utilizados no treino de algum modelo de machine learning. Além do treino, também é possível realizar o tratamento
        dos dados utilizados para a validação final do modelo, são normalmente dados diferentes do treino.

        É recomendado que todas as referências para os dados estáticos sejam adicionadas no mesmo arquivo que a
        implementação específica de pré-processamento, isso pode auxiliar na organização do código pois normalmente alguns
        valores utilizados para pré-processar os dados também são usados em outras implementações.
    """

    def __init__(self):
        """
        Atributos Internos:
            start_pre_processing: Armazena o tempo de início do pré-processamento, em segundos.
            end_pre_processing: Armazena o tempo de término do pré-processamento, em segundos.
        """
        self.start_pre_processing = 0
        self.end_pre_processing = 0

    def get_train_data(self) -> tuple:
        """
            Função que pode ser utilizada para obter o conjunto de dados utilizados no processo de treino do modelo.
            Retorna uma tupla contendo dados para o treino e validação, respectivamente.
        """
        self.start_pre_processing = time.time()
        tuple_data = self._on_execute_train_process()
        self.end_pre_processing = time.time()

        return tuple_data

    @abstractmethod
    def _on_execute_train_process(self) -> tuple:
        """
            Função para uso interno da implementação específica de pré-processamento dos dados utilizados no processo de
            treino do modelo.
        """

    def get_data_additional_validation(self):
        """
            Função que pode ser utilizada para a obtenção dos dados que podem ser utilizados para realizar a validação
            adicional / final do modelo obtido.
        """