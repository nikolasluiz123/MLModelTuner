from typing import TypeVar

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, KFold
from tabulate import tabulate

from scikit_learn.hiper_params_search.common_searcher import Searcher
from scikit_learn.history_manager.common import HistoryManager
from scikit_learn.process_manager.pipeline import Pipeline
from scikit_learn.validator.common_validator import Result

Pipe = TypeVar('Pipe', bound=Pipeline)
History = TypeVar('History', bound=HistoryManager)

class MultiProcessManager:

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
        self.data_x = data_x
        self.data_y = data_y
        self.pipelines = pipelines
        self.history_manager = history_manager
        self.scoring = scoring
        self.save_history = save_history
        self.history_index = history_index

        self.results = []

        np.random.seed(seed)

        if stratified:
            self.cv = StratifiedKFold(n_splits=fold_splits, shuffle=True, random_state=seed)
        else:
            self.cv = KFold(n_splits=fold_splits, shuffle=True, random_state=seed)


    def process_pipelines(self):
        if type(self.pipelines) is list:
            for pipeline in self.pipelines:
                self._process_single_pipeline(pipeline)
        else:
            self._process_single_pipeline(self.pipelines)

        df_results = self._show_results()
        self._on_after_process_pipelines(df_results)

    def _process_single_pipeline(self, pipeline):
        self._process_feature_selection(pipeline)

        search_cv = self._process_hiper_params_search(pipeline)
        validation_result = self._process_validation(pipeline, search_cv)

        self._save_data_in_history(pipeline, validation_result)
        self._append_new_result(pipeline, validation_result)

    def _process_feature_selection(self, pipeline: Pipe):
        if self.history_index is None:
            features =  pipeline.feature_searcher.select_features(
                estimator=pipeline.estimator,
                data_x=self.data_x,
                data_y=self.data_y,
                scoring=self.scoring,
                cv=self.cv
            )

            self.data_x_best_features = features

    def _process_hiper_params_search(self, pipeline: Pipe) -> Searcher | None:
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
        if search_cv is None:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)
        else:
            return pipeline.validator.validate(searcher=search_cv,
                                               data_x=self.data_x_best_features,
                                               data_y=self.data_y,
                                               scoring=self.scoring,
                                               cv=self.cv)

    def _save_data_in_history(self, pipeline: Pipe, result: Result):
        if self.save_history and self.history_index is None:
            feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

            pipeline.history_manager.save_result(result,
                                                 feature_selection_time=self._format_time(feature_selection_time),
                                                 search_time=self._format_time(search_time),
                                                 validation_time=self._format_time(validation_time),
                                                 scoring=self.scoring,
                                                 features=self.data_x_best_features.columns.tolist())

    def _get_execution_times(self, pipeline):
        feature_selection_time = pipeline.feature_searcher.end_search_features_time - pipeline.feature_searcher.start_search_features_time
        search_time = pipeline.params_searcher.end_search_parameter_time - pipeline.params_searcher.start_search_parameter_time
        validation_time = pipeline.validator.end_best_model_validation - pipeline.validator.start_best_model_validation
        return feature_selection_time, search_time, validation_time

    def _append_new_result(self, pipeline: Pipe, result: Result):
        pipeline_infos = pipeline.get_dict_pipeline_data()
        performance_metrics = result.append_data(pipeline_infos)

        if self.history_index is None:
            self._calculate_processes_time(performance_metrics, pipeline)
        else:
            self._load_processes_time_from_history(performance_metrics, pipeline)

        self.results.append(performance_metrics)

    def _calculate_processes_time(self, performance_metrics, pipeline: Pipe):
        feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

        performance_metrics['feature_selection_time'] = self._format_time(feature_selection_time)
        performance_metrics['search_time'] = self._format_time(search_time)
        performance_metrics['validation_time'] = self._format_time(validation_time)

    def _load_processes_time_from_history(self, performance_metrics, pipeline: Pipe):
        history_dict = pipeline.history_manager.get_dictionary_from_json(self.history_index)

        performance_metrics['feature_selection_time'] = history_dict['feature_selection_time']
        performance_metrics['search_time'] = history_dict['search_time']
        performance_metrics['validation_time'] = history_dict['validation_time']

    def _show_results(self) -> DataFrame:
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['mean', 'median', 'standard_deviation'], ascending=False)

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results

    def _on_after_process_pipelines(self, df_results: DataFrame):
        self.__save_best_estimator(df_results)

    def __save_best_estimator(self, df_results: DataFrame):
        if self.save_history and self.history_index is None:
            best = df_results.head(1)

            best_pipeline = self.get_best_pipeline(best)
            validation_result = best_pipeline.history_manager.load_validation_result_from_history()
            dict_history = best_pipeline.history_manager.get_dictionary_from_json(index=-1)

            self.history_manager.save_result(classifier_result=validation_result,
                                             feature_selection_time=best['feature_selection_time'].values[0],
                                             search_time=best['search_time'].values[0],
                                             validation_time=best['validation_time'].values[0],
                                             scoring=best['scoring'].values[0],
                                             features=dict_history['features'].split(','))

    def get_best_pipeline(self, best):
        if type(self.pipelines) is list:
            best_pipeline = [pipe for pipe in self.pipelines if self.__is_best_pipeline(best, pipe)][0]
        else:
            best_pipeline = self.pipelines

        return best_pipeline

    def __is_best_pipeline(self, df: DataFrame, pipe: Pipe):
        return (
                df['estimator'].values[0] == type(pipe.estimator).__name__ and
                df['feature_searcher'].values[0] == type(pipe.feature_searcher).__name__ and
                df['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                df['validator'].values[0] == type(pipe.validator).__name__
        )

    @staticmethod
    def _format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"