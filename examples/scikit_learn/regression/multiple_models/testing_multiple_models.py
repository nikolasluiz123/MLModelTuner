from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skopt.learning import RandomForestRegressor

from examples.scikit_learn.regression.pre_processor import ScikitLearnWorkoutPreProcessorWithScalerExample, \
    ScikitLearnWorkoutPreProcessorExample
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.additional_validator import ScikitLearnRegressorAdditionalValidator
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

########################################################################################################################
#                                    Preparando Implementações que serão Testadas                                      #
########################################################################################################################

pre_processor_with_scaler = ScikitLearnWorkoutPreProcessorWithScalerExample(
    data_path='../data/workout_train_data.csv'
)

pre_processor = ScikitLearnWorkoutPreProcessorExample(
    data_path='../data/workout_train_data.csv'
)

params_searcher = ScikitLearnRandomCVHyperParamsSearcher(number_iterations=100, log_level=1)

history_manager_decision_tree = ScikitLearnCrossValidationHistoryManager(output_directory='history',
                                                           models_directory='models_decision_tree',
                                                           best_params_file_name='params_decision_tree',
                                                           cv_results_file_name='cv_results_decision_tree')

history_manager_random_forest = ScikitLearnCrossValidationHistoryManager(output_directory='history',
                                                           models_directory='models_random_fores',
                                                           best_params_file_name='params_random_fores',
                                                           cv_results_file_name='cv_results_random_fores')

history_manager_kneighbors = ScikitLearnCrossValidationHistoryManager(output_directory='history',
                                                           models_directory='models_kneighbors',
                                                           best_params_file_name='params_kneighbors',
                                                           cv_results_file_name='cv_results_kneighbors')


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
        estimator=DecisionTreeRegressor(),
        params={
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': [None, 'sqrt', 'log2'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager_decision_tree,
        validator=cross_validator
    ),
    ScikitLearnPipeline(
        estimator=RandomForestRegressor(),
        params={
            'n_estimators': randint(10, 50),
            'criterion': ['poisson', 'friedman_mse', 'squared_error', 'absolute_error'],
            'max_depth': randint(1, 20),
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
        },
        data_pre_processor=pre_processor,
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager_random_forest,
        validator=cross_validator
    ),
    ScikitLearnPipeline(
        estimator=KNeighborsRegressor(),
        params={
            'n_neighbors': randint(1, 20),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        data_pre_processor=pre_processor_with_scaler,
        feature_searcher=None,
        params_searcher=params_searcher,
        history_manager=history_manager_kneighbors,
        validator=cross_validator
    ),
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

########################################################################################################################
#                            Realizando Testes Adicionais com o Melhor Modelo Encontrado                               #
########################################################################################################################

best_estimator = best_params_history_manager.get_saved_model(best_params_history_manager.get_history_len())

final_validator = ScikitLearnRegressorAdditionalValidator(
    estimator=best_estimator,
    prefix_file_names='best_estimator',
    validation_results_directory='additional_validations',
    data=pre_processor_with_scaler.get_data_additional_validation(),
    show_graphics=False
)

final_validator.validate()
