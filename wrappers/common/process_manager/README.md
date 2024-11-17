## Módulo Comum de Gerenciamento de Multi Processos

Esse é o módulo principal da biblioteca, ele centraliza todos os processos necessário para
a obtenção do melhor modelo de machine learning.

### Gerenciador de Multi Processos

A implementação [CommonMultiProcessManager]() garante que haja um padrão bem estabelecido
entre as diferentes bibliotecas que devem possuir um MultiProcessManager específico para
executar os pipelines definidos, cada um contendo as configurações específicas de cada processo.

Além da execução dos Pipelines é preciso realizar logs e salvar os dados da execução no histórico,
o MultiProcessManager centraliza tudo isso e permite que cada biblioteca que é utilizada
tenha suas especificidades.

Por padrão o histórico é sempre salvo para cada Pipeline da lista e também existe um histórico
geral, que é controlado pelo MultiProcessManager para guardar o melhor modelo encontrado na
execução realizada de forma separada. Mais informações referentes a manutenção de histórico
podem ser encontradas [nesse readme]()

### Pipeline

Um pipeline consiste basicamente em um conjunto de atributos que definem quais implementações
e valores de parâmetros devem ser utilizados para processar o modelo especificado. Basicamente
um Pipeline pode ser interpretado como um container que guarda tudo que precisa ser executado.

A implementação [CommonPipeline]() é responsável por garantir que os pipelines específicos de cada
biblioteca utilizada internamente contenham comportamentos necessários nos processos do 
MultiProcessManager.
