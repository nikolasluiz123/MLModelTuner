import gc
from abc import ABC

import keras
from pandas import DataFrame
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

from wrappers.common.process_manager.common_process_manager import CommonMultiProcessManager
from wrappers.common.utils.date_time_utils import DateTimeUtils
from wrappers.keras.history_manager.common_history_manager import KerasCommonHistoryManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.results.common_validation_result import KerasValidationResult


class KerasMultiProcessManager(CommonMultiProcessManager[KerasPipeline, KerasCommonHistoryManager, KerasValidationResult], ABC):
    """
    Implementação de gerenciamento dos processos necessário para obtenção do melhor modelo de rede neural implementado
    com keras.
    """

    def __init__(self,
                 pipelines: list[KerasPipeline] | KerasPipeline,
                 history_manager: KerasCommonHistoryManager,
                 save_history=True,
                 history_index: int = None,
                 seed: int = 42,
                 policy='mixed_float16',
                 delete_trials_after_execution=False):
        """
        :param policy: Indica qual deve ser a precisão dos números utilizados nos cálculos das funções da rede neural.
                       Por padrão o valor utilizado é `mixed_float16` por ser mais leve e manter bons resultados.

        :param delete_trials_after_execution: Flag que indica se, ao fim da execução de todos os pipelines devem ser
                                              deletados os arquivos gerados pelo keras na busca de hiperparâmetros que
                                              servem para retomar a execução de busca de hiperparâmetros.

        Atributos Internos:
            train_data: Dados utilizados no treinamento do modelo obtidos através do pré-processamento
            validation_data: Dados utilizados na validação do modelo obtidos através do pré-processamento
        """
        super().__init__(pipelines, history_manager, save_history, history_index, seed)
        self.delete_trials_after_execution = delete_trials_after_execution

        self.train_data = None
        self.validation_data = None

        set_global_policy(policy)
        keras.utils.set_random_seed(seed)

    def _process_single_pipeline(self, pipeline: KerasPipeline):
        self._show_log_init_process(pipeline)

        self.__pre_process_data(pipeline)
        model_instance = self.__process_hyper_params_search(pipeline)
        validation_result = self.__process_validation(pipeline, model_instance)

        self.__save_data_in_history(pipeline, validation_result)
        self._append_new_result(pipeline, validation_result)

        keras.backend.clear_session()
        gc.collect()

    def __pre_process_data(self, pipeline: KerasPipeline):
        """
        Executa o pré-processamento dos dados de acordo com o pipeline. Os dados pré-processados são atribuídos em variáveis
        de acesso global dentro da implementação.

        Somente é executado o pré-processamento dos dados quando não for fornecido `history_index` ou aquele pipeline
        não tiver sido executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
            train_data, validation_data = pipeline.data_pre_processor.get_train_data()

            self.train_data = train_data
            self.validation_data = validation_data

    def __process_hyper_params_search(self, pipeline: KerasPipeline):
        """
        Realiza a busca de hiperparâmetros de acordo com o pipeline.

        Somente é executada a busca dos hiperparâmetros quando não for fornecido `history_index` ou aquele pipeline não
        tiver sido executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
            return pipeline.params_searcher.process(train_data=self.train_data,
                                                    validation_data=self.validation_data,
                                                    model=pipeline.model)
        else:
            return None

    def __process_validation(self, pipeline: KerasPipeline, model_instance) -> KerasValidationResult:
        """
        Realiza a validação do modelo de acordo com o pipeline.

        Somente é executada a validação quando não for fornecido `history_index` ou aquele pipeline não tiver sido
        executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        :param model_instance: Instância do modelo obtida da busca de parâmetros
        """
        if self.history_index is None or not pipeline.history_manager.has_history():
            validation_result = pipeline.validator.validate(
                model_instance=model_instance,
                train_data=self.train_data,
                validation_data=self.validation_data
            )

            return validation_result
        else:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)

    def __save_data_in_history(self, pipeline: KerasPipeline, validation_result: KerasValidationResult):
        """
        Realiza a persistência das informações de execução no arquivo de histórico de acordo com o pipeline.

        Somente serão salvas as informações se `save_history` for True e não for uma execução baseada em dados do histórico
        ou aquele pipeline não ter sido executado ainda (não existir histórico para ser recuperado).

        :param pipeline: Pipeline que está sendo executado
        :param validation_result: Objeto com os dados da validação
        """

        if self.save_history and (self.history_index is None or not pipeline.history_manager.has_history()):
            pre_processing_time, params_search_time, validation_time = pipeline.get_execution_times()

            pipeline.history_manager.save_result(model_instance=validation_result.model,
                                                 model=pipeline.model,
                                                 validation_history=validation_result.history,
                                                 oracle_fields_list=pipeline.params_searcher.get_fields_oracle_json_file(),
                                                 params_search_directory=pipeline.params_searcher.directory,
                                                 params_search_project=pipeline.params_searcher.project_name,
                                                 pre_processing_time=DateTimeUtils.format_time(pre_processing_time),
                                                 params_search_time=DateTimeUtils.format_time(params_search_time),
                                                 validation_time=DateTimeUtils.format_time(validation_time))

    def _calculate_processes_time(self, execution_data_dictionary, pipeline: KerasPipeline):
        pre_processing_time, params_search_time, validation_time = pipeline.get_execution_times()

        execution_data_dictionary['pre_processing_time'] = DateTimeUtils.format_time(pre_processing_time)
        execution_data_dictionary['params_search_time'] = DateTimeUtils.format_time(params_search_time)
        execution_data_dictionary['validation_time'] = DateTimeUtils.format_time(validation_time)

    def _load_processes_time_from_history(self, execution_data_dictionary, pipeline: KerasPipeline):
        history_dict = pipeline.history_manager.get_dictionary_from_params_json(self.history_index)

        execution_data_dictionary['pre_processing_time'] = history_dict['pre_processing_time']
        execution_data_dictionary['params_search_time'] = history_dict['params_search_time']
        execution_data_dictionary['validation_time'] = history_dict['validation_time']


    def _on_after_process_pipelines(self, df_results: DataFrame):
        super()._on_after_process_pipelines(df_results)
        self.__delete_all_pipeline_trials()

    def __delete_all_pipeline_trials(self):
        """
        Realiza a remoção dos dados de histórico das tentativas de busca de hiperparâmetros utilizando as implementações
        do keras.

        Os dados históricos do keras só serão removidos se o usuário informar isso através da flag `delete_trials_after_execution`.
        """
        if self.delete_trials_after_execution:
            if type(self.pipelines) is list:
                for pipeline in self.pipelines:
                    pipeline.history_manager.delete_trials(pipeline.params_searcher.directory,
                                                           pipeline.params_searcher.project_name)
            else:
                self.pipelines.history_manager.delete_trials(self.pipelines.params_searcher.directory,
                                                             self.pipelines.params_searcher.project_name)

    def _save_best_model(self, df_results: DataFrame):
        pipeline_not_executed = self._get_has_pipeline_not_executed()

        if self.save_history and (self.history_index is None or pipeline_not_executed):
            best = df_results.head(1)

            best_pipeline = self._get_best_pipeline(best)
            dict_history = best_pipeline.history_manager.get_dictionary_from_params_json(index=-1)
            model_instance = best_pipeline.history_manager.get_saved_model(best_pipeline.history_manager.get_history_len())

            self.history_manager.save_result(model_instance=model_instance,
                                             model=best_pipeline.model,
                                             validation_history=dict_history,
                                             params_search_directory=best_pipeline.params_searcher.directory,
                                             params_search_project=best_pipeline.params_searcher.project_name,
                                             oracle_fields_list=best_pipeline.params_searcher.get_fields_oracle_json_file(),
                                             pre_processing_time=dict_history['pre_processing_time'],
                                             params_search_time=dict_history['params_search_time'],
                                             validation_time=dict_history['validation_time'])

    def _is_best_pipeline(self, df_results: DataFrame, pipe: KerasPipeline):
        return (
                df_results['model'].values[0] == type(pipe.model).__name__ and
                df_results['data_pre_processor'].values[0] == type(pipe.data_pre_processor).__name__ and
                df_results['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df_results['searcher_objective'].values[0] == pipe.params_searcher.objective and
                df_results['searcher_epochs'].values[0] == pipe.params_searcher.epochs and
                df_results['searcher_batch_size'].values[0] == pipe.params_searcher.batch_size and
                df_results['project_name'].values[0] == pipe.params_searcher.project_name and
                df_results['validator'].values[0] == type(pipe.validator).__name__ and
                df_results['validator_epochs'].values[0] == pipe.validator.epochs and
                df_results['validator_batch_size'].values[0] == pipe.validator.batch_size
        )
