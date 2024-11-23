## Módulo de Histórico das Execuções

É possível ver detalhes referentes a implementação comum de manutenção do histórico acessando [esse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/history_manager/README.md).

Em relação ao keras, existem implementações de histórico implementadas internamente no processo
de busca de hiperparâmetros, nesses processos internos o keras armazena as tentativas executadas
organizando elas em projetos. Com esse tipo de implementação, se você tiver executado 25 tentativas e
um erro ocorrer ou você precisar desligar a máquina, ao iniciar o processo novamente ele começará
da tentativa 26 (supondo que a 25 tenha terminado).

Sabendo disso foi preciso implementar um módulo de histórico específico para lidar com essa peculiaridade
da biblioteca keras, em resumo, uma parte dos dados que o gerenciador vai salvar vem da leitura
de um arquivo chamado ``oracle.json`` que é onde o keras mantém os dados do melhor modelo.

Por padrão, depois que todos os pipelines são executados os dados da tentativas que estavam armazenados
na pasta designada são deletados, o objetivo é não manter arquivos que são relativamente grandes
(principalmente os arquivos de pesos da rede neural) sendo que a ideia é sempre reproduzir uma execução
baseada no gerenciador de histórico implementado aqui.

Uma última peculiaridade que pode ser encontrada nesse gerenciador de histórico implementado é que
precisamos implementar um gerenciador específico para classificação, isso ocorre devido ao objeto
de retorno da validação ser diferente para classificação e regressão.

Para facilitar a implementação de classificação e regressão foi criada a classe [KerasCommonHistoryManager](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/history_manager/common_history_manager.py#L14)
que centraliza todos os comportamentos comuns do keras em relação a histórico.

Como implementação específica de classificação temos [KerasClassifierHistoryManager](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/history_manager/classifier_history_manager.py#L6) e para regressão
temos [KerasRegressorHistoryManager](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/history_manager/regressor_history_manager.py#L6).
