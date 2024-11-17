## Módulo de Busca dos Melhores Parâmetros dos Modelos

É possível ver detalhes referentes a implementação comum acessando [esse readme]().

Esse módulo é focado em implementações de busca dos hiper parâmetros dos modelos testados.
Abaixo serão apresentados brevemente os tipos de busca implementados e será referenciada a documentação
para garantir um melhor entendimento.

### Implementação Comum para Scikit-Learn

A classe [ScikitLearnCommonHyperParamsSearcher]() é a base para a implementação de todos os wrappers de busca de parâmetros,
ela define tudo que é normalmente utilizado e utiliza de valores padrões para os cenários
excepcionais que não utilizem algo.

### Buscas Random

Um tipo de implementação presente no [scikit-learn](https://scikit-learn.org/stable/)
é baseado em explorar apenas uma parte das possibilidades de valores dos parâmetros que
desejar. É bastante comum que seja solicitado como um dos parâmetros o número máximo de 
iterações, além disso, parâmetros de scoring e validação cruzada são muito importantes
e também costumam estar presentes.

Abaixo segue a relação das implementações criadas e o que é utilizado internamente: 

| Implementação Externa                          | Implementação Wrapper                                                                                                                                                | 
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [RandomizedSearchCV]()                         | [ScikitLearnRandomCVHyperParamsSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/random_searcher.py#L10)        | 
| [HalvingRandomSearchCV]()                      | [ScikitLearnHalvingRandomCVHyperParamsSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/random_searcher.py#L34) |

Em relação a passagem de valores e parâmetros para ser testados ambas as implementações
acima aceitam as seguintes declarações:

```
params = {
    'nome_param_1': [1, 2, 3, 4], 
    'nome_param_2': True,
    'nome_param_3': randint(1, 10)
    'nome_param_4': uniform(loc=0.1, scale=0.4)
}
```

Acima vemos a declaração de um dicionário python contendo os parâmetros e valores desejados
para a realizar a busca do melhor modelo. Veja que é possível passar uma lista de valores,
um valor único ou a chamada de alguma função que retorna um número inteiro ou decimal dentro
de um intervalo.

### Buscas em Grid

Esse tipo de busca explora todas as combinações possíveis para os valores dos parâmetros especificados.
Essa é a principal diferença em relação a Busca Random, por conta disso o processamento costuma
levar mais tempo, em compensação todas as possibilidades serão testadas, basta escolher com cuidado
quais parâmetros e valores explorar.

Abaixo segue a relação das implementações criadas e o que é utilizado internamente:

| Implementação Externa                          | Implementação Wrapper                                                                                                                                            | 
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [GridSearchCV]()                               | [ScikitLearnGridCVHyperParamsSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/grid_searcher.py#L8)         | 
| [HalvingGridSearchCV]()                        | [ScikitLearnHalvingGridCVHyperParamsSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/grid_searcher.py#L29) |

Em relação a passagem de valores e parâmetros para ser testados ambas as implementações
acima aceitam apenas declarações de valor único ou uma lista de valores:

```
params = {
    'nome_param_1': [1, 2, 3, 4], 
    'nome_param_2': True
}
```

Como se trata da busca em uma Grid só são aceitos valores finitos.

### Busca Bayesiana

Esse tipo de busca não está dentro da biblioteca do scikit-learn, mas sim no [scikit-optimize](https://scikit-optimize.github.io/stable/).
Em relação a busca em grid ele pode ser mais performático pois utiliza o processo gaussiano
para prever quais combinações de valores dos parâmetros podem ser mais promissoras.

Para a busca de parâmetros além do scikit-learn também foi utilizada a biblioteca .

| Implementação Externa                          | Implementação Wrapper                                                                                                                                        | 
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [BayesSearchCV]()                              | [ScikitLearnBayesianHyperParamsSearcher](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/bayesian_search.py#L8) |

Essa implementação é um pouco diferente em relação as demais para a passagem de valores
para os parâmetros que deseja testar:

```
params = {
    'nome_param_1': (1, 10), 
    'nome_param_2': (0.5, 0.9)
}
```

```
params = {
    'nome_param_1': Real(1e-6, 1e+6, prior='log-uniform'), 
    'nome_param_2': Integer(1,8),
    'nome_param_3': Categorical(['category_a', 'category_b', 'category_c'])
}
```

Acima temos as duas possibilidades de passagem de valores para os parâmetros, a primeira
abordagem é mais simples e pode ser utilizada para valores numéricos, sendo baseada em
tuplas e os valores informados são sempre mínimo e máximo. A outra possibilidade é utilizar
classes da biblioteca scikit-optimizer que representam dados de números inteiros, decimais ou dados categóricos.
