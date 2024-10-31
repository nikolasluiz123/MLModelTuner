from scikit_learn.history_manager.common import HistoryManager
from scikit_learn.validator.results.cross_validation import CrossValidationResult


class CrossValidatorHistoryManager(HistoryManager[CrossValidationResult]):

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str):
        super().__init__(output_directory, models_directory, params_file_name)

    def save_result(self,
                    classifier_result: CrossValidationResult,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        dictionary = {
            'estimator': type(classifier_result.estimator).__name__,
            'mean': classifier_result.mean,
            'standard_deviation': classifier_result.standard_deviation,
            'median': classifier_result.median,
            'variance': classifier_result.variance,
            'standard_error': classifier_result.standard_error,
            'min_max_score': classifier_result.min_max_score,
            'estimator_params': classifier_result.estimator.get_params(),
            'scoring': scoring,
            'features': ", ".join(features),
            'feature_selection_time': feature_selection_time,
            'search_time': search_time,
            'validation_time': validation_time
        }

        self._create_output_dir()
        self._save_dictionary_in_json(dictionary)
        self._save_model(classifier_result.estimator)

    def load_validation_result_from_history(self, index: int = -1) -> CrossValidationResult:
        result_dict = self.get_dictionary_from_json(index)

        return CrossValidationResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            scoring=result_dict['scoring'],
            estimator=self.get_saved_model(self._get_history_len()),
        )
