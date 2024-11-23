## Módulo do Keras

Esse módulo visa referenciar todas as implementações para a busca dos melhores modelos implementados 
como redes neurais com o auxílio do [tensorflow](https://www.tensorflow.org/api_docs/python/tf) e [keras](https://keras.io/api/).

### Pré-Processamento dos Dados

Antes de realizar qualquer processo precisamos tratar e organizar os dados que serão utilizados,
seja lá de onde eles forem obtidos. Internamente essa é o primeiro passo executado e ele é detalhado
dentro [desse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/data_pre_processor/README.md).

### Busca dos Melhores Parâmetros do Modelo

Utilizando a biblioteca [keras-tuner](https://keras.io/keras_tuner/) é possível testar diversos valores
de muitos parâmetros que podem ser passados para as camadas da rede neural.

Cada uma das implementações presentes estão mais detalhadas dentro [desse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/hyper_params_search/README.md).

### Validação do Modelo

Para termos certeza de que o modelo é realmente o melhor possível precisamos adotar alguma das
maneiras de validação existentes, infelizmente não fornece nenhuma implementação específica para
a validação de uma rede neural, mas, é possível realizar algumas implementações.

Essas implementações são detalhadas dentro [desse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/validator/README.md).

### Armazenamento dos Dados de Processamento

Algo que é crucial quando se está procurando pelo melhor modelo para alguma finalidade é conseguir
olhar para trás e ver quais foram os resultados obtidos em todas as execuções realizadas, a biblioteca
fornece essa possibilidade através de implementações chamadas de HistoryManagers os quais são
mais detalhados [nesse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/history_manager/README.md).

### Processamento dos Modelos

Uma das funcionalidades mais importantes e que centraliza tudo são os chamados ProcessManagers
que estão implementados na biblioteca, em resumo, eles centralizam o uso de todas as implementações
citadas anteriormente e, através de pipelines, realizam as execuções para cada modelo desejado.

É possível verificar detalhes sobre os ProcessManagers e Pipelines [nesse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/process_manager/README.md)
