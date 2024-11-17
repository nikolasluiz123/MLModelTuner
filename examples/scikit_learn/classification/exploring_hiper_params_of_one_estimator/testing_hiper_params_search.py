import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer

from examples.scikit_learn.classification.pre_processor import ScikitLearnTitanicPreProcessorExample
from wrappers.scikit_learn.features_search.select_k_best_searcher import ScikitLearnSelectKBestSearcher
from wrappers.scikit_learn.hyper_params_search.bayesian_search import ScikitLearnBayesianHyperParamsSearcher
from wrappers.scikit_learn.hyper_params_search.grid_searcher import ScikitLearnGridCVHyperParamsSearcher, \
    ScikitLearnHalvingGridCVHyperParamsSearcher
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher, \
    ScikitLearnHalvingRandomCVHyperParamsSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

########################################################################################################################
#                                    Preparando Comuns para o Modelo Testado                                           #
########################################################################################################################

pre_processor = ScikitLearnTitanicPreProcessorExample()

feature_searcher = ScikitLearnSelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
validator = ScikitLearnCrossValidator(log_level=1)

history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history',
                                                           models_directory='decision_tree_classifier_models',
                                                           best_params_file_name='decision_tree_classifier_best_params',
                                                           cv_results_file_name='decision_tree_classifier_cv_results')

best_params_history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history_bests',
                                                                       models_directory='best_models',
                                                                       best_params_file_name='best_params',
                                                                       cv_results_file_name='best_params_cv_results')

########################################################################################################################
#                                               Criando os Pipelines                                                   #
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
        params_searcher=ScikitLearnRandomCVHyperParamsSearcher(number_iterations=50, log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=ScikitLearnHalvingRandomCVHyperParamsSearcher(log_level=1, number_candidates=50),
        validator=validator,
        history_manager=history_manager
    ),
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=ScikitLearnGridCVHyperParamsSearcher(log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [2, 4, 8, 16],
            'min_weight_fraction_leaf': [0.1, 0.3, 0.4],
            'max_features': [None, 'sqrt', 'log2'],
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=ScikitLearnHalvingGridCVHyperParamsSearcher(log_level=1),
        validator=validator,
        history_manager=history_manager
    ),
    ScikitLearnPipeline(
        estimator=DecisionTreeClassifier(),
        params={
            'criterion': Categorical(['gini', 'entropy', 'log_loss']),
            'splitter': Categorical(['best', 'random']),
            'max_depth': Integer(1, 10),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(2, 20),
            'min_weight_fraction_leaf': Real(0.1, 0.5, prior='uniform'),
            'max_features': Categorical([None, 'sqrt', 'log2']),
        },
        data_pre_processor=pre_processor,
        feature_searcher=feature_searcher,
        params_searcher=ScikitLearnBayesianHyperParamsSearcher(log_level=1, number_iterations=50),
        validator=validator,
        history_manager=history_manager
    ),
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
    history_index=None,
    stratified=True,
)

manager.process_pipelines()
