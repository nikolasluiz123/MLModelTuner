from typing import TypeVar

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate

from wrappers.scikit_learn.hiper_params_search.common_searcher import Searcher
from wrappers.scikit_learn.history_manager.common import HistoryManager
from wrappers.scikit_learn.process_manager.pipeline import Pipeline
from wrappers.scikit_learn.validator.common_validator import Result

Pipe = TypeVar('Pipe', bound=Pipeline)
History = TypeVar('History', bound=HistoryManager)


class MultiProcessManager:
    """
    Gerencia a execução de múltiplos pipelines de machine learning em um processo de validação cruzada.

    Esta classe executa múltiplos pipelines de modelagem, permitindo a busca de hiperparâmetros,
    seleção de features e validação do modelo. Os resultados são armazenados em um gerenciador de histórico.

    :param data_x: Conjunto de dados de características (features) para treinamento e validação.
    :param data_y: Conjunto de dados de rótulos (labels) correspondentes a data_x.
    :param seed: Semente para a randomização do processo de validação cruzada.
    :param fold_splits: Número de divisões (folds) para a validação cruzada.
    :param pipelines: Um ou mais pipelines que serão processados.
    :param history_manager: Gerenciador de histórico para armazenar resultados.
    :param scoring: Métrica a ser utilizada para validação do modelo.
    :param stratified: Indica se a validação cruzada deve ser estratificada (default: False).
    :param save_history: Indica se os resultados devem ser salvos no histórico (default: True).
    :param history_index: Índice do histórico a ser carregado, se aplicável (default: None).
    """

    def __init__(self,
                 data_x,
                 data_y,
                 seed: int,
                 fold_splits: int,
                 pipelines: list[Pipe] | Pipe,
                 history_manager: History,
                 scoring: str,
                 stratified: bool = False,
                 save_history: bool = True,
                 history_index: int = None):
        """
        Inicializa o MultiProcessManager.

        :param data_x: Conjunto de dados de características (features).
        :param data_y: Conjunto de dados de rótulos (labels).
        :param seed: Semente para a randomização.
        :param fold_splits: Número de folds para a validação cruzada.
        :param pipelines: Um ou mais pipelines a serem processados.
        :param history_manager: Gerenciador de histórico para armazenar resultados.
        :param scoring: Métrica a ser utilizada para validação.
        :param stratified: Indica se a validação cruzada deve ser estratificada.
        :param save_history: Indica se os resultados devem ser salvos no histórico.
        :param history_index: Índice do histórico a ser carregado, se aplicável.
        """
        self.data_x = data_x
        self.data_y = data_y
        self.pipelines = pipelines
        self.history_manager = history_manager
        self.scoring = scoring
        self.save_history = save_history
        self.history_index = history_index

        self.results = []
        self.data_x_scaled = None

        np.random.seed(seed)

        if stratified:
            self.cv = StratifiedKFold(n_splits=fold_splits, shuffle=True, random_state=seed)
        else:
            self.cv = KFold(n_splits=fold_splits, shuffle=True, random_state=seed)

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

    def _process_single_pipeline(self, pipeline):
        """
        Processa um único pipeline, incluindo seleção de features, busca de hiperparâmetros
        e validação.

        :param pipeline: O pipeline a ser processado.
        """
        self.__show_log_init_process(pipeline)
        self.__scale_data(pipeline)
        self._process_feature_selection(pipeline)

        search_cv = self._process_hiper_params_search(pipeline)
        validation_result = self._process_validation(pipeline, search_cv)

        self._save_data_in_history(pipeline, validation_result, search_cv)
        self._append_new_result(pipeline, validation_result)

    def __show_log_init_process(self, pipeline):
        if self.history_index is None:
            print()
            print('Iniciando o Processamento')
            data = pipeline.get_dict_pipeline_data()
            data = {k: [v] for k, v in data.items()}
            df = pd.DataFrame.from_dict(data, orient='columns')
            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print()

    def __scale_data(self, pipeline: Pipe):
        if pipeline.scaler is not None:
            self.data_x_scaled = pipeline.scaler.fit_transform(self.data_x)

    def __get_dataframe_from_scaled_data(self):
        if self.data_x_scaled is not None:
            return pd.DataFrame(self.data_x_scaled, columns=self.data_x.columns)
        else:
            return self.data_x

    def _process_feature_selection(self, pipeline: Pipe):
        """
        Realiza a seleção de features usando o pipeline especificado.

        :param pipeline: O pipeline que contém a lógica de seleção de features.
        """
        if self.history_index is None:
            if pipeline.feature_searcher is None:
                self.data_x_best_features = self.data_x
            else:
                features = pipeline.feature_searcher.select_features(
                    estimator=pipeline.estimator,
                    data_x=self.__get_dataframe_from_scaled_data(),
                    data_y=self.data_y,
                    scoring=self.scoring,
                    cv=self.cv
                )

                self.data_x_best_features = features

    def _process_hiper_params_search(self, pipeline: Pipe) -> Searcher | None:
        """
        Realiza a busca de hiperparâmetros para o pipeline especificado, se não houver histórico a ser carregado.

        :param pipeline: O pipeline que contém a lógica de busca de hiperparâmetros.
        :return: O objeto Searcher resultante da busca, ou None se estiver carregando do histórico.
        """
        if self.history_index is None:
            return pipeline.params_searcher.search_hiper_parameters(
                estimator=pipeline.estimator,
                params=pipeline.params,
                data_x=self.data_x_best_features,
                data_y=self.data_y,
                scoring=self.scoring,
                cv=self.cv
            )
        else:
            return None

    def _process_validation(self, pipeline: Pipe, search_cv: Searcher) -> Result:
        """
        Valida o pipeline usando os dados e a lógica de validação apropriados.

        :param pipeline: O pipeline a ser validado.
        :param search_cv: O objeto Searcher resultante da busca de hiperparâmetros.
        """
        if search_cv is None:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)
        else:
            return pipeline.validator.validate(searcher=search_cv,
                                               data_x=self.data_x_best_features,
                                               data_y=self.data_y,
                                               scoring=self.scoring,
                                               cv=self.cv)

    def _save_data_in_history(self, pipeline: Pipe, result: Result, searcher: Searcher):
        """
        Salva os resultados da validação no gerenciador de histórico, se a opção de salvar estiver habilitada.

        :param pipeline: O pipeline que gerou os resultados.
        :param result: O resultado da validação a ser salvo.
        """
        if self.save_history and self.history_index is None:
            feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

            pipeline.history_manager.save_result(result,
                                                 feature_selection_time=self._format_time(feature_selection_time),
                                                 search_time=self._format_time(search_time),
                                                 validation_time=self._format_time(validation_time),
                                                 scoring=self.scoring,
                                                 features=self.data_x_best_features.columns.tolist(),
                                                 cv_results=searcher.cv_results_,
                                                 scaler=pipeline.scaler)

    def _get_execution_times(self, pipeline):
        """
        Obtém os tempos de execução das fases de seleção de features, busca de hiperparâmetros e validação.

        :param pipeline: O pipeline que contém as informações de tempo.
        :return: Tupla com os tempos de seleção de features, busca e validação.
        """
        if pipeline.feature_searcher is not None:
            feature_selection_time = pipeline.feature_searcher.end_search_features_time - pipeline.feature_searcher.start_search_features_time
        else:
            feature_selection_time = 0.0

        search_time = pipeline.params_searcher.end_search_parameter_time - pipeline.params_searcher.start_search_parameter_time
        validation_time = pipeline.validator.end_best_model_validation - pipeline.validator.start_best_model_validation

        return feature_selection_time, search_time, validation_time

    def _append_new_result(self, pipeline: Pipe, result: Result):
        """
        Anexa o novo resultado da validação aos resultados do pipeline.

        :param pipeline: O pipeline cujos resultados estão sendo anexados.
        :param result: O resultado da validação a ser anexado.
        """
        pipeline_infos = pipeline.get_dict_pipeline_data()
        performance_metrics = result.append_data(pipeline_infos)

        if self.history_index is None:
            self._calculate_processes_time(performance_metrics, pipeline)
        else:
            self._load_processes_time_from_history(performance_metrics, pipeline)

        self.results.append(performance_metrics)

    def _calculate_processes_time(self, performance_metrics, pipeline: Pipe):
        """
        Calcula e formata os tempos de execução e os adiciona às métricas de desempenho.

        :param performance_metrics: Dicionário contendo as métricas de desempenho.
        :param pipeline: O pipeline que contém as informações de tempo.
        """
        feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

        performance_metrics['feature_selection_time'] = self._format_time(feature_selection_time)
        performance_metrics['search_time'] = self._format_time(search_time)
        performance_metrics['validation_time'] = self._format_time(validation_time)

    def _load_processes_time_from_history(self, performance_metrics, pipeline: Pipe):
        """
        Carrega os tempos de execução do histórico e os adiciona às métricas de desempenho.

        :param performance_metrics: Dicionário contendo as métricas de desempenho.
        :param pipeline: O pipeline cujas informações estão sendo carregadas.
        """
        history_dict = pipeline.history_manager.get_dictionary_from_params_json(self.history_index)

        performance_metrics['feature_selection_time'] = history_dict['feature_selection_time']
        performance_metrics['search_time'] = history_dict['search_time']
        performance_metrics['validation_time'] = history_dict['validation_time']

    def _show_results(self) -> DataFrame:
        """
        Exibe os resultados em um formato tabular e retorna um DataFrame dos resultados.

        :return: DataFrame contendo os resultados dos pipelines processados.
        """
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['mean', 'median', 'standard_deviation'], ascending=False)

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results

    def _on_after_process_pipelines(self, df_results: DataFrame):
        """
        Executa ações após o processamento de todos os pipelines, como salvar o melhor estimador.

        :param df_results: DataFrame contendo os resultados dos pipelines.
        """
        self.__save_best_estimator(df_results)

    def __save_best_estimator(self, df_results: DataFrame):
        """
        Salva o melhor estimador no gerenciador de histórico, se a opção de salvar estiver habilitada.

        :param df_results: DataFrame contendo os resultados dos pipelines.
        """
        if self.save_history and self.history_index is None:
            best = df_results.head(1)

            best_pipeline = self.get_best_pipeline(best)
            validation_result = best_pipeline.history_manager.load_validation_result_from_history()
            dict_history = best_pipeline.history_manager.get_dictionary_from_params_json(index=-1)
            dict_cv_results = best_pipeline.history_manager.get_dictionary_from_cv_results_json(index=-1)

            self.history_manager.save_result(classifier_result=validation_result,
                                             feature_selection_time=best['feature_selection_time'].values[0],
                                             search_time=best['search_time'].values[0],
                                             validation_time=best['validation_time'].values[0],
                                             scoring=best['scoring'].values[0],
                                             features=dict_history['features'].split(','),
                                             cv_results=dict_cv_results,
                                             scaler=dict_history['scaler'])

    def get_best_pipeline(self, best):
        """
        Obtém o pipeline que teve o melhor desempenho com base nos resultados.

        :param best: DataFrame contendo os melhores resultados.
        :return: O pipeline correspondente ao melhor desempenho.
        """
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self.__is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def __is_best_pipeline(self, df: DataFrame, pipe: Pipe):
        """
        Verifica se o pipeline é o melhor com base nos resultados.

        :param df: DataFrame contendo os melhores resultados.
        :param pipe: O pipeline a ser verificado.
        :return: True se o pipeline for o melhor, False caso contrário.
        """
        return (
                df['estimator'].values[0] == type(pipe.estimator).__name__ and
                df['feature_searcher'].values[0] == type(pipe.feature_searcher).__name__ and
                df['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df['validator'].values[0] == type(pipe.validator).__name__
        )

    @staticmethod
    def _format_time(seconds):
        """
        Formata o tempo em segundos para o formato HH:MM:SS.mmm.

        :param seconds: Tempo em segundos a ser formatado.
        :return: String formatada representando o tempo.
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"
