from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeClassifier

from examples.scikit_learn.classification.pre_processor import ScikitLearnTitanicPreProcessorExample
from wrappers.scikit_learn.features_search.rfe_searcher import ScikitLearnRecursiveFeatureCVSearcher
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

########################################################################################################################
#                                    Preparando Implementações que serão Testadas                                      #
########################################################################################################################

data_pre_processor = ScikitLearnTitanicPreProcessorExample()

recursive_feature_cv_searcher = ScikitLearnRecursiveFeatureCVSearcher(log_level=1)

random_cv_hyper_params_searcher = ScikitLearnRandomCVHyperParamsSearcher(number_iterations=500, log_level=1)

cross_validator_history_manager = ScikitLearnCrossValidationHistoryManager(output_directory='history',
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

pipeline = ScikitLearnPipeline(
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
    data_pre_processor=data_pre_processor,
    feature_searcher=recursive_feature_cv_searcher,
    params_searcher=random_cv_hyper_params_searcher,
    history_manager=cross_validator_history_manager,
    validator=cross_validator
)

########################################################################################################################
#                                      Criando e Executando o Process Manager                                          #
########################################################################################################################

manager = ScikitLearnMultiProcessManager(
    pipelines=pipeline,
    history_manager=best_params_history_manager,
    fold_splits=5,
    scoring='accuracy',
    save_history=True,
    history_index=None,
    stratified=True,
)

manager.process_pipelines()
