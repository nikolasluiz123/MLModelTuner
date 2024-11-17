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

    def __init__(self,
                 pipelines: list[Pipeline] | Pipeline,
                 history_manager: HistoryManager,
                 save_history: bool = True,
                 history_index: int = None,
                 seed: int = 42):
        self.pipelines = pipelines
        self.history_manager = history_manager
        self.save_history = save_history
        self.history_index = history_index
        self.seed = seed

        self.results = []

    @abstractmethod
    def _process_single_pipeline(self, pipeline: Pipeline):
        ...

    @abstractmethod
    def _show_results(self):
        ...

    @abstractmethod
    def _save_best_model(self, df_results: DataFrame):
        ...

    @abstractmethod
    def _is_best_pipeline(self, df: DataFrame, pipe: Pipeline):
        ...

    @abstractmethod
    def _calculate_processes_time(self, validation_result_dictionary: dict, pipeline: Pipeline):
        ...

    @abstractmethod
    def _load_processes_time_from_history(self, validation_result_dictionary: dict, pipeline: Pipeline):
        ...

    def process_pipelines(self):
        """
        Processa todos os pipelines especificados, realizando seleção de features,
        busca de hiperparâmetros e validação.

        Os resultados são apresentados em formato tabular e salvos no histórico, se aplicável.
        """
        if type(self.pipelines) is list:
            for pipeline in self.pipelines:
                self._process_single_pipeline(pipeline)
        else:
            self._process_single_pipeline(self.pipelines)

        df_results = self._show_results()
        self._on_after_process_pipelines(df_results)

    def _on_after_process_pipelines(self, df_results: DataFrame):
        self._save_best_model(df_results)

    def _get_has_pipeline_not_executed(self):
        pipeline_not_executed = False

        if type(self.pipelines) is list:
            for p in self.pipelines:
                if not p.history_manager.has_history():
                    pipeline_not_executed = True
        else:
            if not self.pipelines.history_manager.has_history():
                pipeline_not_executed = True

        return pipeline_not_executed

    def _get_best_pipeline(self, best):
        """
        Obtém o pipeline que teve o melhor desempenho com base nos resultados.

        :param best: DataFrame contendo os melhores resultados.
        :return: O pipeline correspondente ao melhor desempenho.
        """
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self._is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def _show_log_init_process(self, pipeline: Pipeline):
        if self.history_index is None or not pipeline.history_manager.has_history():
            print()
            print('Iniciando o Processamento')

            data = pipeline.get_dict_pipeline_data()
            data = {k: [v] for k, v in data.items()}
            df = pd.DataFrame.from_dict(data, orient='columns')

            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print()

    def _append_new_result(self, pipeline: Pipeline, result: CommonValResult):
        """
        Anexa o novo resultado da validação aos resultados do pipeline.

        :param pipeline: O pipeline cujos resultados estão sendo anexados.
        :param result: O resultado da validação a ser anexado.
        """
        pipeline_dictionary = pipeline.get_dict_pipeline_data()
        validation_result_dictionary = result.append_data(pipeline_dictionary)

        if self.history_index is None or not pipeline.history_manager.has_history():
            self._calculate_processes_time(validation_result_dictionary, pipeline)
        else:
            self._load_processes_time_from_history(validation_result_dictionary, pipeline)

        self.results.append(validation_result_dictionary)

