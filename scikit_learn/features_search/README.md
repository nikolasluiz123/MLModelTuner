## Módulo de Busca das Melhores Features

Existem alguns tipos de implementações presentes no scikit-learn, apenas algumas dessas implementações
foram utilizadas para essa biblioteca e essas implementações serão detalhadas abaixo referenciando
sempre a documentação do próprio scikit-learn para garantir total entendimento.

### CommonFeaturesSearcher

A classe [CommonFeaturesSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/common_searcher.py#L4)
é utilizada como classe base para a implementação de todas as classes wrapper de seleção de features.
Ela considera os parâmetros que são normalmente utilizados pelas implementações do scikit-learn se beneficiando
na maioria das vezes da definição de valores padrões para que seja possível utilizar de uma melhor forma
implementações diferentes que não necessitam dos mesmos parâmetros.

### Relação das Implementações do Scikit-Learn Utilizadas

| Implementação do Scikit-Learn                                                                                                         | Implementação Wrapper                                                                                                                                                | 
|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)                           | [SelectKBestSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/select_k_best_searcher.py#L8)                         | 
| [SelectPercentile](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html)                 | [SelectPercentileSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/select_percentile_searcher.py#L8)                |
| [GenericUnivariateSelect](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html)   | [GenericUnivariateSelectSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/generic_univariate_select_searcher.py#L8) | 
| [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)                                           | [RecursiveFeatureSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/rfe_searcher.py#L8)                              |
| [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)                                       | [RecursiveFeatureCVSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/rfe_searcher.py#L34)                           |
