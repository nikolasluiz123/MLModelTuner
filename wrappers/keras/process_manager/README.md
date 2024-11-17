## Módulo de Gerenciamento de Multi Processos

É possível ver detalhes referentes a implementação comum dos gerenciadores acessando [esse readme]().

A implementação especifica do keras é [KerasMultiProcessManager]() que concentra todos os processos
específicos dele. Nessa implementação existem alguns processos específicos da biblioteca que são
necessários.

O pipeline é definido por [KerasPipeline]() pois para redes neurais com keras temos classes específicas para cada um
dos processos.

A cada nova execução de um pipeline, precisamos realizar a limpeza da memória para tentar evitar
problemas gerados por falta de recursos, existem também alguns valores parametrizáveis que são
específicos para processamento de redes neurais e, por fim, o keras mantém uma implementação
de histórico específica para a busca de hiperparâmetros, o qual devemos manupular tanto para obter
informações quanto para eliminar quando não for mais necessário.