import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate

from wrappers.common.process_manager.common_process_manager import CommonMultiProcessManager
from wrappers.common.utils.date_time_utils import DateTimeUtils
from wrappers.scikit_learn.hiper_params_search.common_hyper_params_searcher import ScikitLearnSearcher
from wrappers.scikit_learn.history_manager.common_history_manager import ScikitLearnCommonHistoryManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.results.cross_validation_result import ScikitLearnCrossValidationResult


class ScikitLearnMultiProcessManager(CommonMultiProcessManager[ScikitLearnPipeline, ScikitLearnCommonHistoryManager, ScikitLearnCrossValidationResult]):
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
                 pipelines: list[ScikitLearnPipeline] | ScikitLearnPipeline,
                 history_manager: ScikitLearnCommonHistoryManager,
                 fold_splits: int,
                 scoring: str,
                 save_history: bool = True,
                 history_index: int = None,
                 seed: int = 42,
                 stratified: bool = False):
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
        super().__init__(pipelines, history_manager, save_history, history_index, seed)
        self.scoring = scoring
        self.data_x = None
        self.data_y = None
        self.data_x_scaled = None

        np.random.seed(seed)

        if stratified:
            self.cv = StratifiedKFold(n_splits=fold_splits, shuffle=True, random_state=seed)
        else:
            self.cv = KFold(n_splits=fold_splits, shuffle=True, random_state=seed)

    def _process_single_pipeline(self, pipeline):
        """
        Processa um único pipeline, incluindo seleção de features, busca de hiperparâmetros
        e validação.

        :param pipeline: O pipeline a ser processado.
        """
        self._show_log_init_process(pipeline)
        self.__pre_process_data(pipeline)
        self.__scale_data(pipeline)
        self._process_feature_selection(pipeline)

        search_cv = self._process_hyper_params_search(pipeline)
        validation_result = self._process_validation(pipeline, search_cv)

        self._save_data_in_history(pipeline, validation_result, search_cv)
        self._append_new_result(pipeline, validation_result)

    def __pre_process_data(self, pipeline: ScikitLearnPipeline):
        if self.history_index is None or not pipeline.history_manager.has_history():
            data_x, data_y = pipeline.data_pre_processor.get_train_data()

            self.data_x = data_x
            self.data_y = data_y

    def __scale_data(self, pipeline: ScikitLearnPipeline):
        if pipeline.scaler is not None:
            self.data_x_scaled = pipeline.scaler.fit_transform(self.data_x)

    def __get_dataframe_from_scaled_data(self):
        if self.data_x_scaled is not None:
            return pd.DataFrame(self.data_x_scaled, columns=self.data_x.columns)
        else:
            return self.data_x

    def _process_feature_selection(self, pipeline: ScikitLearnPipeline):
        """
        Realiza a seleção de features usando o pipeline especificado.

        :param pipeline: O pipeline que contém a lógica de seleção de features.
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
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

    def _process_hyper_params_search(self, pipeline: ScikitLearnPipeline) -> ScikitLearnSearcher | None:
        """
        Realiza a busca de hiperparâmetros para o pipeline especificado, se não houver histórico a ser carregado.

        :param pipeline: O pipeline que contém a lógica de busca de hiperparâmetros.
        :return: O objeto Searcher resultante da busca, ou None se estiver carregando do histórico.
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
            return pipeline.params_searcher.search_hyper_parameters(
                estimator=pipeline.estimator,
                params=pipeline.params,
                data_x=self.data_x_best_features,
                data_y=self.data_y,
                scoring=self.scoring,
                cv=self.cv
            )
        else:
            return None

    def _process_validation(self, pipeline: ScikitLearnPipeline, search_cv: ScikitLearnSearcher) -> ScikitLearnCrossValidationResult:
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

    def _save_data_in_history(self,
                              pipeline: ScikitLearnPipeline,
                              result: ScikitLearnCrossValidationResult,
                              searcher: ScikitLearnSearcher):
        """
        Salva os resultados da validação no gerenciador de histórico, se a opção de salvar estiver habilitada.

        :param pipeline: O pipeline que gerou os resultados.
        :param result: O resultado da validação a ser salvo.
        """
        if self.save_history and (self.history_index is None or not pipeline.history_manager.has_history()):
            pre_processing_time, feature_selection_time, search_time, validation_time = pipeline.get_execution_times()

            pipeline.history_manager.save_result(result,
                                                 pre_processing_time=DateTimeUtils.format_time(feature_selection_time),
                                                 feature_selection_time=DateTimeUtils.format_time(feature_selection_time),
                                                 search_time=DateTimeUtils.format_time(search_time),
                                                 validation_time=DateTimeUtils.format_time(validation_time),
                                                 scoring=self.scoring,
                                                 features=self.data_x_best_features.columns.tolist(),
                                                 cv_results=searcher.cv_results_,
                                                 scaler=pipeline.scaler)

    def _calculate_processes_time(self, validation_result_dictionary: dict, pipeline: ScikitLearnPipeline):
        """
        Calcula e formata os tempos de execução e os adiciona às métricas de desempenho.

        :param validation_result_dictionary: Dicionário contendo as métricas de desempenho.
        :param pipeline: O pipeline que contém as informações de tempo.
        """
        pre_processing_time, feature_selection_time, search_time, validation_time = pipeline.get_execution_times()

        validation_result_dictionary['pre_processing_time'] = DateTimeUtils.format_time(pre_processing_time)
        validation_result_dictionary['feature_selection_time'] = DateTimeUtils.format_time(feature_selection_time)
        validation_result_dictionary['search_time'] = DateTimeUtils.format_time(search_time)
        validation_result_dictionary['validation_time'] = DateTimeUtils.format_time(validation_time)

    def _load_processes_time_from_history(self, validation_result_dictionary: dict, pipeline: ScikitLearnPipeline):
        """
        Carrega os tempos de execução do histórico e os adiciona às métricas de desempenho.

        :param validation_result_dictionary: Dicionário contendo as métricas de desempenho.
        :param pipeline: O pipeline cujas informações estão sendo carregadas.
        """
        history_dict = pipeline.history_manager.get_dictionary_from_params_json(self.history_index)

        validation_result_dictionary['pre_processing_time'] = history_dict['pre_processing_time']
        validation_result_dictionary['feature_selection_time'] = history_dict['feature_selection_time']
        validation_result_dictionary['search_time'] = history_dict['feature_selection_time']
        validation_result_dictionary['validation_time'] = history_dict['feature_selection_time']

    def _show_results(self) -> DataFrame:
        """
        Exibe os resultados em um formato tabular e retorna um DataFrame dos resultados.

        :return: DataFrame contendo os resultados dos pipelines processados.
        """
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['mean', 'median', 'standard_deviation'], ascending=[False, False, True])

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results

    def _save_best_model(self, df_results: DataFrame):
        """
        Salva o melhor estimador no gerenciador de histórico, se a opção de salvar estiver habilitada.

        :param df_results: DataFrame contendo os resultados dos pipelines.
        """
        pipeline_not_executed = self._get_has_pipeline_not_executed()

        if self.save_history and (self.history_index is None or pipeline_not_executed):
            best = df_results.head(1)

            best_pipeline = self._get_best_pipeline(best)
            validation_result = best_pipeline.history_manager.load_validation_result_from_history()
            dict_history = best_pipeline.history_manager.get_dictionary_from_params_json(index=-1)
            dict_cv_results = best_pipeline.history_manager.get_dictionary_from_cv_results_json(index=-1)

            self.history_manager.save_result(validation_result=validation_result,
                                             pre_processing_time=best['pre_processing_time'].values[0],
                                             feature_selection_time=best['feature_selection_time'].values[0],
                                             search_time=best['search_time'].values[0],
                                             validation_time=best['validation_time'].values[0],
                                             scoring=best['scoring'].values[0],
                                             features=dict_history['features'].split(','),
                                             cv_results=dict_cv_results,
                                             scaler=dict_history['scaler'])


    def _is_best_pipeline(self, df: DataFrame, pipe: ScikitLearnPipeline):
        """
        Verifica se o pipeline é o melhor com base nos resultados.

        :param df: DataFrame contendo os melhores resultados.
        :param pipe: O pipeline a ser verificado.
        :return: True se o pipeline for o melhor, False caso contrário.
        """
        return (
                df['estimator'].values[0] == type(pipe.estimator).__name__ and
                df['data_pre_processor'].values[0] == type(pipe.data_pre_processor).__name__ and
                df['feature_searcher'].values[0] == type(pipe.feature_searcher).__name__ and
                df['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df['validator'].values[0] == type(pipe.validator).__name__
        )
