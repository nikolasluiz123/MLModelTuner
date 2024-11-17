import gc
from abc import ABC

import keras
from pandas import DataFrame
from tensorflow.python.keras.mixed_precision.policy import set_global_policy

from wrappers.common.process_manager.common_process_manager import CommonMultiProcessManager
from wrappers.common.utils.date_time_utils import DateTimeUtils
from wrappers.keras.history_manager.common_history_manager import KerasHistoryManager
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.validator.results.common_validation_result import KerasValidationResult


class KerasMultiProcessManager(CommonMultiProcessManager[KerasPipeline, KerasHistoryManager, KerasValidationResult], ABC):

    def __init__(self,
                 pipelines: list[KerasPipeline] | KerasPipeline,
                 history_manager: KerasHistoryManager,
                 save_history=True,
                 history_index: int = None,
                 seed: int = 42,
                 policy='mixed_float16',
                 delete_trials_after_execution=False):
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

    def __process_hyper_params_search(self, pipeline: KerasPipeline):
        if self.history_index is None:
            return pipeline.params_searcher.process(train_data=self.train_data,
                                                    validation_data=self.validation_data,
                                                    model=pipeline.model)
        else:
            return None

    def __process_validation(self, pipeline: KerasPipeline, model_instance) -> KerasValidationResult:
        if model_instance is not None:
            validation_result = pipeline.validator.validate(
                model_instance=model_instance,
                train_data=self.train_data,
                validation_data=self.validation_data
            )

            return validation_result
        else:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)

    def __pre_process_data(self, pipeline: KerasPipeline):
        if self.history_index is None or not pipeline.history_manager.has_history():
            train_data, validation_data = pipeline.data_pre_processor.get_train_data()

            self.train_data = train_data
            self.validation_data = validation_data

    def __save_data_in_history(self, pipeline: KerasPipeline, validation_result: KerasValidationResult):
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

    def _calculate_processes_time(self, validation_result_dictionary, pipeline: KerasPipeline):
        pre_processing_time, params_search_time, validation_time = pipeline.get_execution_times()

        validation_result_dictionary['pre_processing_time'] = DateTimeUtils.format_time(pre_processing_time)
        validation_result_dictionary['params_search_time'] = DateTimeUtils.format_time(params_search_time)
        validation_result_dictionary['validation_time'] = DateTimeUtils.format_time(validation_time)

    def _load_processes_time_from_history(self, validation_result_dictionary, pipeline: KerasPipeline):
        history_dict = pipeline.history_manager.get_dictionary_from_params_json(self.history_index)

        validation_result_dictionary['pre_processing_time'] = history_dict['pre_processing_time']
        validation_result_dictionary['params_search_time'] = history_dict['params_search_time']
        validation_result_dictionary['validation_time'] = history_dict['validation_time']


    def _on_after_process_pipelines(self, df_results: DataFrame):
        super()._on_after_process_pipelines(df_results)
        self.__delete_all_pipeline_trials()

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

    def _is_best_pipeline(self, df: DataFrame, pipe: KerasPipeline):
        return (
                df['model'].values[0] == type(pipe.model).__name__ and
                df['data_pre_processor'].values[0] == type(pipe.data_pre_processor).__name__ and
                df['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df['searcher_objective'].values[0] == pipe.params_searcher.objective and
                df['searcher_epochs'].values[0] == pipe.params_searcher.epochs and
                df['searcher_batch_size'].values[0] == pipe.params_searcher.batch_size and
                df['project_name'].values[0] == pipe.params_searcher.project_name and
                df['validator'].values[0] == type(pipe.validator).__name__ and
                df['validator_epochs'].values[0] == pipe.validator.epochs and
                df['validator_batch_size'].values[0] == pipe.validator.batch_size
        )

    def __delete_all_pipeline_trials(self):
        if self.history_index is None and self.delete_trials_after_execution:
            if type(self.pipelines) is list:
                for pipeline in self.pipelines:
                    pipeline.history_manager.delete_trials(pipeline.params_searcher.directory,
                                                           pipeline.params_searcher.project_name)
            else:
                self.pipelines.history_manager.delete_trials(self.pipelines.params_searcher.directory,
                                                             self.pipelines.params_searcher.project_name)
