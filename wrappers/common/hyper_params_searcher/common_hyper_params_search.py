from abc import ABC


class CommonHyperParamsSearch(ABC):
    """
    Implementação responsável por conter aquilo que é comum entre as buscas de hiperparâmetros das diferentes bibliotecas
    utilizadas.
    """

    def __init__(self, log_level: int = 0):
        """
        :param log_level: Inteiro que representa o quanto de informação será exibida no console durante a execução
                          da busca de hiperparâmetros.

        Atributos Internos:
            start_search_parameter_time: Armazena o tempo de início da busca de hiperparâmetros, em segundos.
            end_search_parameter_time: Armazena o tempo de término da busca de hiperparâmetros, em segundos.
        """
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0
