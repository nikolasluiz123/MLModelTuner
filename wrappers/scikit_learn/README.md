## Módulo do Scikit Learn

Esse módulo tem como objetivo referenciar todas as implementações para a busca dos melhores 
modelos existentes dentro do [scikit-learn](https://scikit-learn.org/stable/).

### Busca das Features

O scikit-learn fornece diversas implementações de busca das melhores features que podem ser
utilizadas para classificação ou regressão, sabendo disso, foram implementadas algumas classes
wrapper que utilizam implementações do scikit-learn para a busca das melhores features e evitar
uma análise manual.

Cada uma das implementações presentes estão mais detalhadas dentro [desse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/features_search/README.md).

### Busca dos Melhores Parâmetros do Modelo

Uma das partes cruciais para obter bons resultados com um modelo de machine learning é 
descobrir quais os melhores parâmetros para o modelo e os dados que você está utilizando. A
biblioteca também fornece implementações wrapper para utilizar algumas variações de busca o
que vai possibilitar experimentar a que melhor se encaixa.

Cada uma das implementações presentes estão mais detalhadas dentro [desse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/hiper_params_search/README.md).

### Validação do Modelo

Para termos certeza de que o modelo é realmente o melhor possível precisamos adotar alguma das
maneiras de validação existentes, a biblioteca possui algumas implementações de validação as
quais detalhamos [nesse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/validator/README.md).

### Armazenamento dos Dados de Processamento

Algo que é crucial quando se está procurando pelo melhor modelo para alguma finalidade é conseguir
olhar para trás e ver quais foram os resultados obtidos em todas as execuções realizadas, a biblioteca
fornece essa possibilidade através de implementações chamadas de HistoryManagers os quais são
mais detalhados [nesse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/history_manager/README.md).

### Processamento dos Modelos

Uma das funcionalidades mais importantes e que centraliza tudo são os chamados ProcessManagers
que estão implementados na biblioteca, em resumo, eles centralizam o uso de todas as implementações
citadas anteriormente e, através de pipelines, realizam as execuções para cada modelo desejado.

É possível verificar detalhes sobre os ProcessManagers e Pipelines [nesse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/process_manager/README.md)
