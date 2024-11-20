from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from examples.scikit_learn.classification.pre_processor import ScikitLearnTitanicPreProcessorExample
from wrappers.scikit_learn.features_search.select_k_best_searcher import ScikitLearnSelectKBestSearcher
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.classifier_additional_validator import ScikitLearnClassifierAdditionalValidator
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

########################################################################################################################
#                                    Preparando Comuns para os Modelos Testados                                        #
########################################################################################################################
pre_processor = ScikitLearnTitanicPreProcessorExample()

feature_searcher = ScikitLearnSelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
params_searcher = ScikitLearnRandomCVHyperParamsSearcher(number_iterations=50, log_level=1)
validator = ScikitLearnCrossValidator(log_level=1)

best_params_history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history_bests',
                                                                       models_directory='best_models',
                                                                       best_params_file_name='best_params',
                                                                       cv_results_file_name='best_params_cv_results')

########################################################################################################################
#                           Criando os Pipelines com os Modelos e suas Especificidades                                 #
########################################################################################################################

pipelines = [
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=ScikitLearnCrossValidationHistoryManager(
            output_directory='history',
            models_directory='decision_tree_classifier_models',
            best_params_file_name='decision_tree_classifier_best_params',
            cv_results_file_name='decision_tree_classifier_cv_results',
        )
    ),
    ScikitLearnPipeline(
        estimator=RandomForestClassifier(),
        params={
            'n_estimators': randint(10, 50),
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2']
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=ScikitLearnCrossValidationHistoryManager(
            output_directory='history',
            models_directory='random_forest_classifier_models',
            best_params_file_name='random_forest_classifier_best_params',
            cv_results_file_name='random_forest_classifier_cv_results',
        )
    ),
    ScikitLearnPipeline(
        estimator=GaussianProcessClassifier(),
        params={
            'optimizer': ['fmin_l_bfgs_b', None],
            'n_restarts_optimizer': randint(0, 10),
            'multi_class': ['one_vs_rest', 'one_vs_one'],
            'max_iter_predict': [100, 300, 500, 700, 900]
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=ScikitLearnCrossValidationHistoryManager(
            output_directory='history',
            models_directory='gausian_process_classifier_models',
            best_params_file_name='gausian_process_classifier_best_params',
            cv_results_file_name='gausian_process_classifier_cv_results',
        )
    ),
    ScikitLearnPipeline(
        estimator=KNeighborsClassifier(),
        params={
            'n_neighbors': randint(1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(1, 100),
            'p': [1, 2],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=validator,
        history_manager=ScikitLearnCrossValidationHistoryManager(
            output_directory='history',
            models_directory='k_neighbors_classifier_models',
            best_params_file_name='k_neighbors_classifier_best_params',
            cv_results_file_name='k_neighbors_classifier_cv_results',
        )
    )
]

########################################################################################################################
#                                      Criando e Executando o Process Manager                                          #
########################################################################################################################

manager = ScikitLearnMultiProcessManager(
    pipelines=pipelines,
    history_manager=best_params_history_manager,
    fold_splits=5,
    scoring='accuracy',
    save_history=True,
    history_index=-1,
    stratified=True,
)

manager.process_pipelines()

########################################################################################################################
#                            Realizando Testes Adicionais com o Melhor Modelo Encontrado                               #
########################################################################################################################

best_estimator = best_params_history_manager.load_validation_result_from_history().estimator
final_validator = ScikitLearnClassifierAdditionalValidator(
    estimator=best_estimator,
    prefix_file_names='best_estimator',
    validation_results_directory='additional_validations',
    data=pre_processor.get_data_additional_validation()
)
final_validator.validate()
