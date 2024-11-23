import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate

from wrappers.common.process_manager.common_process_manager import CommonMultiProcessManager
from wrappers.common.utils.date_time_utils import DateTimeUtils
from wrappers.scikit_learn.hyper_params_search.common_hyper_params_searcher import ScikitLearnSearcher
from wrappers.scikit_learn.history_manager.common_history_manager import ScikitLearnCommonHistoryManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.results.cross_validation_result import ScikitLearnCrossValidationResult


class ScikitLearnMultiProcessManager(CommonMultiProcessManager[ScikitLearnPipeline, ScikitLearnCommonHistoryManager, ScikitLearnCrossValidationResult]):
    """
    Implementação para gerenciamento de processos necessário para obtenção do melhor modelo do scikit-learn.
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
        :param scoring: Métrica a ser utilizada para validação.
        :param fold_splits: Número de folds para a validação cruzada.
        :param stratified: Indica se a validação cruzada deve ser estratificada.

        Atributos Internos:
            data_x: Dados das features obtidos a partir do pré-processamento

            data_y: Dados do target obtidos a partir do pré-processamento

            data_x_scaled: Dados das features após aplicar um scaler. Pode ser None caso não desejar realizar esse processo.

            data_x_best_features: Dados das features que são melhores para o modelo específico que está rodando no momento.
            Pode ser None caso não desejar realizar esse processo.

            cv: Implementação de Fold para validação cruzada
        """

        super().__init__(pipelines, history_manager, save_history, history_index, seed)
        self.scoring = scoring
        self.data_x = None
        self.data_y = None
        self.data_x_best_features = None

        np.random.seed(seed)

        if stratified:
            self.cv = StratifiedKFold(n_splits=fold_splits, shuffle=True, random_state=seed)
        else:
            self.cv = KFold(n_splits=fold_splits, shuffle=True, random_state=seed)

    def _process_single_pipeline(self, pipeline):
        self._show_log_init_process(pipeline)
        self.__pre_process_data(pipeline)
        self.__process_feature_selection(pipeline)

        search_cv = self.__process_hyper_params_search(pipeline)
        validation_result = self.__process_validation(pipeline, search_cv)

        self.__save_data_in_history(pipeline, validation_result, search_cv)
        self._append_new_result(pipeline, validation_result)

    def __pre_process_data(self, pipeline: ScikitLearnPipeline):
        """
        Executa o pré-processamento dos dados de acordo com o pipeline. Os dados pré-processados são atribuídos em variáveis
        de acesso global dentro da implementação.

        Somente é executado o pré-processamento dos dados quando não for fornecido `history_index` ou aquele pipeline
        não tiver sido executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        """

        if self.history_index is None or not pipeline.history_manager.has_history():
            data_x, data_y = pipeline.data_pre_processor.get_train_data()

            self.data_x = data_x
            self.data_y = data_y

    def __process_feature_selection(self, pipeline: ScikitLearnPipeline):
        """
        Realiza a seleção das melhores features a partir dos dados presentes em `data_x`.

        Somente é executado esse processo quando não for fornecido `history_index` ou aquele pipeline
        não tiver sido executado ainda (não existir histórico para ser recuperado).

        Além disso, a seleção das melhores features é um processo opcional, se não for definida uma implementação no
        pipeline esse processo não será realizado.

        :param pipeline: Pipeline que está sendo executado
        """

        if self.history_index is None or not pipeline.history_manager.has_history():
            if pipeline.feature_searcher is None:
                self.data_x_best_features = self.data_x
            else:
                features = pipeline.feature_searcher.select_features(
                    estimator=pipeline.estimator,
                    data_x=self.data_x,
                    data_y=self.data_y,
                    scoring=self.scoring,
                    cv=self.cv
                )

                self.data_x_best_features = features

    def __process_hyper_params_search(self, pipeline: ScikitLearnPipeline) -> ScikitLearnSearcher | None:
        """
        Realiza a busca de hiperparâmetros de acordo com o pipeline.

        Somente é executada a busca dos hiperparâmetros quando não for fornecido `history_index` ou aquele pipeline não
        tiver sido executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
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

    def __process_validation(self, pipeline: ScikitLearnPipeline, search_cv: ScikitLearnSearcher) -> ScikitLearnCrossValidationResult:
        """
        Realiza a validação do modelo de acordo com o pipeline.

        Somente é executada a validação quando não for fornecido `history_index` ou aquele pipeline não tiver sido
        executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        :param search_cv: Retorno da busca dos melhores hiperparâmetros
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
            return pipeline.validator.validate(searcher=search_cv,
                                               data_x=self.data_x_best_features,
                                               data_y=self.data_y,
                                               scoring=self.scoring,
                                               cv=self.cv)
        else:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)

    def __save_data_in_history(self,
                               pipeline: ScikitLearnPipeline,
                               result: ScikitLearnCrossValidationResult,
                               searcher: ScikitLearnSearcher):
        """
        Realiza a persistência das informações de execução no arquivo de histórico de acordo com o pipeline.

        Somente serão salvas as informações se `save_history` for True e não for uma execução baseada em dados do histórico
        ou aquele pipeline não ter sido executado ainda (não existir histórico para ser recuperado).

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
                                                 cv_results=searcher.cv_results_)

    def _calculate_processes_time(self, execution_data_dictionary: dict, pipeline: ScikitLearnPipeline):
        pre_processing_time, feature_selection_time, search_time, validation_time = pipeline.get_execution_times()

        execution_data_dictionary['pre_processing_time'] = DateTimeUtils.format_time(pre_processing_time)
        execution_data_dictionary['feature_selection_time'] = DateTimeUtils.format_time(feature_selection_time)
        execution_data_dictionary['search_time'] = DateTimeUtils.format_time(search_time)
        execution_data_dictionary['validation_time'] = DateTimeUtils.format_time(validation_time)

    def _load_processes_time_from_history(self, execution_data_dictionary: dict, pipeline: ScikitLearnPipeline):
        history_dict = pipeline.history_manager.get_dictionary_from_params_json(self.history_index)

        execution_data_dictionary['pre_processing_time'] = history_dict['pre_processing_time']
        execution_data_dictionary['feature_selection_time'] = history_dict['feature_selection_time']
        execution_data_dictionary['search_time'] = history_dict['feature_selection_time']
        execution_data_dictionary['validation_time'] = history_dict['feature_selection_time']

    def _show_results(self) -> DataFrame:
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['mean', 'median', 'standard_deviation'], ascending=[False, False, True])

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results

    def _save_best_model(self, df_results: DataFrame):
        if self.save_history:
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
                                             cv_results=dict_cv_results)


    def _is_best_pipeline(self, df_results: DataFrame, pipe: ScikitLearnPipeline):
        return (
                df_results['estimator'].values[0] == type(pipe.estimator).__name__ and
                df_results['data_pre_processor'].values[0] == type(pipe.data_pre_processor).__name__ and
                df_results['feature_searcher'].values[0] == type(pipe.feature_searcher).__name__ and
                df_results['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df_results['validator'].values[0] == type(pipe.validator).__name__
        )
