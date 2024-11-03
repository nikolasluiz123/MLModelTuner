# Introdução

Essa biblioteca python tem como finalidade unir todas as bibliotecas existentes de
Machine Learning em uma única implementação modularizada a qual auxiliará desenvolvedores
que desejam buscar pelo melhor algoritmo que resolva o problema que desejarem, sendo possível testar diferentes tipos de modelos presentes em uma mesma biblioteca utilizando
diferentes implementações nos processos padrões de avaliação de modelos.

# Bibliotecas Suportadas

Abaixo serão declaradas as bibliotecas as quais damos suporte e possuimos implementações
que conseguem facilitar a comparação de performance entre modelos e encontrar o melhor.

## Scikit-Learn

Essa biblioteca extremamente poderosa é utilizada em um de nossos módulos e temos uma
docuentação que pode ser acessada [aqui](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/README.md)
caso seja de seu interesse entender o que cada um dos módulos específicos faz de forma mais superficial,
se desejar entender com o máximo de detalhes convido você a adentrar no código fonte do projeto.

### Alguns Exemplos de Utilização

Se você quer apenas entender na prática como a biblioteca pode auxiliar em seu projeto, abaixo
serão listados alguns exemplos de uso com códigos simples para demonstração.

### Testar um único modelo

É possível que você esteja apenas querendo explorar um modelo que achou interessante ou que
você já sabe que ele vai lidar bem com sua base de dados e o padrão deles ou você está começando
e quer começar do jeito mais simples. Sabendo disso, [veja esse exemplo](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/classification/one_estimator/testing_one_estimator.py).

### Testar um único modelo com diferentes implementações

A biblioteca permite que você utilize um modelo específico e, criando diferentes Pipelines,
defina, por exemplo, diferentes implementações de busca das melhores features para esse modelo,
com a intenção de encontrar qual é a melhor estratégia. Sabendo disso, [veja esse exemplo testando diferentes seleções de features](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/classification/feature_selection_for_one_estimator/testing_feature_selection_for_one_estimator.py)
e entenda como isso pode ser feito.

Um outro exemplo que segue a mesma linha é o caso de você desejar testar diferentes implementações
de busca de parâmetros do modelo, [veja esse exemplo](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/classification/exploring_hiper_params_of_one_estimator/testing_hiper_params_search.py).

### Testando Vários Modelos

Pode ser que você queira explorar a gama de possibilidades disponíveis no scikit-learn,
para isso, a biblioteca possibilita a passagem de Pipelines com diferentes modelos que
serão validados um a um em sequência, ao fim será retornado um DataFrame com N modelos,
apenas a melhor versão encontrada de cada um.

Veja [esse exemplo](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/classification/many_estimators/testing_search_best_estimator.py)
para que possa entender melhor.

Além de testar diferentes modelos, você pode querer realizar algum tipo de validação adicional
apenas com o melhor dos melhores encontrado, isso também é possível, [veja esse exemplo](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/classification/additional_validation/testing_models_with_additional_validation.py).

### Escalando Dados

Alguns modelos, dependendo dos dados, podem se beneficiar de um escalonamento, a biblioteca permite que seja passada
uma implementação de forma opcional ao pipeline para que possa ser comparado se o escalonamento dos dados auxiliaria o
modelo que está sendo testado ou não.

[Esse exemplo](https://github.com/nikolasluiz123/MLModelTunner/blob/master/examples/scikit_learn/regression/scaling_data/testing_scaling_data.py)
mostra que pode ser obtida uma diferença entre dados escalados e não escalados, mesmo que pequena. A grandeza da mudança
vai depender do modelo utilizado e em como são seus dados.

## XGBoost

Futuramente...

## StatsModels

Futuramente...

