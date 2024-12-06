from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

from wrappers.common.history_manager.common_history_manager import CommonHistoryManager, CommonValResult
from wrappers.common.process_manager.common_pipeline import CommonPipeline

Pipeline = TypeVar('Pipeline', bound=CommonPipeline)
HistoryManager = TypeVar('HistoryManager', bound=CommonHistoryManager)

class CommonMultiProcessManager(ABC, Generic[Pipeline, HistoryManager, CommonValResult]):
    """
    Implementação base que contem comportamentos comuns que todos os MultiProcessManagers precisam.

    A ideia de um MultiProcessManager é executar os Pipelines passados utilizando as implementações específicas definidas
    em cada um deles em uma ordem pré-definida, imutável e que faça sentido.

    Por questões de diferenças entre as bibliotecas as quais foram feitos os wrappers a definição dos passos executados
    em um pipeline deve ser implementado em cada extensão de MultiProcessManager.
    """
    def __init__(self,
                 pipelines: list[Pipeline] | Pipeline,
                 history_manager: HistoryManager,
                 save_history: bool = True,
                 history_index: int = None,
                 seed: int = 42):
        """
        :param pipelines: Pipelines que serão executados, podendo ser uma lista ou um único pipeline. Deve obrigatoriamente
                          ser uma implementação de :class:`wrappers.common.process_manager.common_pipeline.CommonPipeline`.

        :param history_manager: Implementação de manutenção do histórico da execuções responsável nesse caso pelos melhores
                                modelos encontrados em cada execução dos pipelines. Deve ser obrigatoriamente uma
                                implementação de :class:`wrappers.common.history_manager.common_history_manager.CommonHistoryManager`.

        :param save_history: Flag que indica se durante a execução dos pipelines os dados devem ser salvos no histórico.

        :param history_index: Inteiro que indica qual o índice da lista de execuções do histórico deseja recuperar para
                              visualizar. Por padrão o valor é None para que não seja recuperado nada do histórico e
                              seja realizada uma nova execução. Os índices podem ser passados como uma lista python comum,
                              a lista aramazenada no JSON será processada e o índice solicitado será retornado.

        :param seed: Inteiro que representa a seed que permite reprodutibilidade entre as diferentes execuções, esse
                     valor é utilizado internamente em vários processos.
        """

        self.pipelines = pipelines
        self.history_manager = history_manager
        self.save_history = save_history
        self.history_index = history_index
        self.seed = seed

        self.results = []

    @abstractmethod
    def _process_single_pipeline(self, pipeline: Pipeline):
        """
        Função responsável por executar os processos necessários na execução do Pipeline, a ordem de execução e o que
        será executado varia de acordo com a biblioteca que a implementação específica está tratando.
        """

    @abstractmethod
    def _show_results(self):
        """
        Função responsável por exibir os dados armazenados na lista results normalmente em forma tabular e com uma ordenação
        que faça sentido, mantendo obrigatoriamente o melhor modelo no topo da lista pois isso será utilizado para salvar
        o melhor modelo encontrado separadamente.
        """

    @abstractmethod
    def _save_best_model(self, df_results: DataFrame):
        """
        Função responsável por salvar os dados do pipeline que resultou no melhor modelo. Devem ser recuperados todos
        os pipelines e identificado o que gerou o melhor modelo utilizando os dados do DataFrame parâmetro df_results que
        foi previamente ordenado e o melhor pipeline sempre será o primeiro registro desse DataFrame.

        O processo especifico pode variar um pouco entre as implementações, mas sempre consistirá em salvar os dados da
        execução e o modelo de machine learning treinado para reutilização.

        :param df_results: DataFrame contendo os resultados das execuções devidamente ordenados do melhor para o pior
        """

    @abstractmethod
    def _is_best_pipeline(self, df_results: DataFrame, pipe: Pipeline) -> bool:
        """
        Função que deve realizar uma comparação entre os dados do DataFrame e os dados do Pipeline para determinar se
        o pipeline em questão foi quem gerou o melhor modelo.

        Basicamente recupera o primeiro registro do DataFrame que já vai estar ordenado corretamente e compara as
        informações. Cada implementação pode possuir informações específicas em seu pipeline, por isso essa implementação
        deve ser específica.

        :param df_results: DataFrame contendo os resultados das execuções devidamente ordenados do melhor para o pior
        :param pipe: Pipeline executado que deseja verificar se foi quem gerou o melhor modelo
        """

    @abstractmethod
    def _calculate_processes_time(self, execution_data_dictionary: dict, pipeline: Pipeline):
        """
        Função que realiza o cálculo dos tempos de processamento, o qual se baseia simplesmente em subtrair o tempo final
        do tempo inicial de cada processo que o pipeline possuir. Essa função só precisa ser utilizada em casos onde não
        deseja recuperar os dados do histórico de execução.

        Cada implementação específica precisa implementar a função por conta dos diferentes processos que podem ser
        executados.

        :param execution_data_dictionary: Dicionário que armazena os dados do pipeline juntamente dos dados do resultado
                                          da validação

        :param pipeline: Pipeline executado que deseja calcular os tempos de processamento
        """

    @abstractmethod
    def _load_processes_time_from_history(self, execution_data_dictionary: dict, pipeline: Pipeline):
        """
        Função que realiza o carregamento dos tempos de processamento do histórico, obtidos da execução do pipeline em
        algum momento.

        Cada implementação específica precisa implementar a função por conta dos diferentes processos que podem ser
        executados.

        :param execution_data_dictionary: Dicionário que armazena os dados do pipeline juntamente dos dados do resultado
                                          da validação

        :param pipeline: Pipeline executado que deseja calcular os tempos de processamento
        """

    def process_pipelines(self):
        """
        Processa todos os pipelines especificados realizando os processos definidos.

        Se for uma lista de pipelines, realizará a iteração e executará cada um deles, um após o outro.
        Se for um único pipeline, executará somente ele.

        Sempre que toda a execução for finalizada os resultados dos melhores modelos serão exibidos de forma tabular, além
        disso, podem ser realizados processamentos adicionais ao fim de todas as execuções. Um processo que é padrão e
        deve ser implementado por todos é salvar o melhor modelo encontrado.
        """
        if type(self.pipelines) is list:
            for pipeline in self.pipelines:
                self._process_single_pipeline(pipeline)
        else:
            self._process_single_pipeline(self.pipelines)

        df_results = self._show_results()
        self._on_after_process_pipelines(df_results)

    def _on_after_process_pipelines(self, df_results: DataFrame):
        """
        Função que realiza processamentos após finalizar a execução de todos os pipelines. Por padrão o melhor modelo
        precisa ser salvo nesse ponto, mas essa função pode ser complementada.
        """
        self._save_best_model(df_results)

    def _get_has_pipeline_not_executed(self) -> bool:
        """
        Função responsável por retornar se existe algum pipeline que ainda não foi executado.

        Essa função é útil nos momentos em que é definida uma lista de pipelines, o processo é executado e o usuário
        adiciona um novo pipeline, mas deseja utilizar os dados históricos dos pipelines já executados.
        """
        pipeline_not_executed = False

        if type(self.pipelines) is list:
            for p in self.pipelines:
                if not p.history_manager.has_history():
                    pipeline_not_executed = True
        else:
            if not self.pipelines.history_manager.has_history():
                pipeline_not_executed = True

        return pipeline_not_executed

    def _get_best_pipeline(self, best: DataFrame) -> Pipeline:
        """
        Retorna o pipeline que teve o melhor desempenho com base nos resultados.

        :param best: DataFrame contendo um registro que representa o melhor resultado.
        """
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self._is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def _show_log_init_process(self, pipeline: Pipeline):
        """
        Função que exibe um log de início de processamento do pipeline específico de forma tabular, contendo os dados
        obtidos em forma de dicionário do pipeline.

        :param pipeline: Pipeline que vai ser executado
        """
        execute = self._get_execute_pipeline(pipeline)

        if execute:
            print()
            print('Iniciando o Processamento')

            data = pipeline.get_dictionary_pipeline_data()
            data = {k: [v] for k, v in data.items()}
            df = pd.DataFrame.from_dict(data, orient='columns')

            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print()

    def _get_execute_pipeline(self, pipeline):
        return (self.history_index is None or
                not pipeline.history_manager.has_history() or
                self.history_index > pipeline.history_manager.get_history_len() - 1)

    def _append_new_result(self, pipeline: Pipeline, result: CommonValResult):
        """
        Anexa o novo resultado da validação aos resultados do pipeline.

        :param pipeline: O pipeline cujos resultados estão sendo anexados.
        :param result: O resultado da validação a ser anexado.
        """
        pipeline_dictionary = pipeline.get_dictionary_pipeline_data()
        execution_data_dictionary = result.append_data(pipeline_dictionary)

        execute = self._get_execute_pipeline(pipeline)

        if execute:
            self._calculate_processes_time(execution_data_dictionary, pipeline)
        else:
            self._load_processes_time_from_history(execution_data_dictionary, pipeline)

        self.results.append(execution_data_dictionary)

