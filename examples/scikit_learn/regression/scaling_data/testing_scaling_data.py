from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from examples.scikit_learn.regression.pre_processor import ScikitLearnWorkoutPreProcessorExample
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

########################################################################################################################
#                                    Preparando Implementações que serão Testadas                                      #
########################################################################################################################

pre_processor = ScikitLearnWorkoutPreProcessorExample()

params_searcher = ScikitLearnRandomCVHyperParamsSearcher(number_iterations=100, log_level=1)

history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history',
                                                           models_directory='models',
                                                           best_params_file_name='params',
                                                           cv_results_file_name='cv_results')

best_params_history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history_bests',
                                                                       models_directory='best_models',
                                                                       best_params_file_name='best_params',
                                                                       cv_results_file_name='best_cv_results')
cross_validator = ScikitLearnCrossValidator(log_level=1)

########################################################################################################################
#                                               Criando o Pipeline                                                     #
########################################################################################################################

pipelines = [
    ScikitLearnPipeline(
        estimator=KNeighborsRegressor(),
        params={
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        data_pre_processor=pre_processor,
        scaler=StandardScaler(),
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager,
        validator=cross_validator
    ),
    ScikitLearnPipeline(
        estimator=KNeighborsRegressor(),
        params={
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager,
        validator=cross_validator
    )
]

########################################################################################################################
#                                      Criando e Executando o Process Manager                                          #
########################################################################################################################

manager = ScikitLearnMultiProcessManager(
    pipelines=pipelines,
    history_manager=best_params_history_manager,
    fold_splits=5,
    scoring='neg_mean_squared_error',
    save_history=True,
    history_index=None
)

manager.process_pipelines()
