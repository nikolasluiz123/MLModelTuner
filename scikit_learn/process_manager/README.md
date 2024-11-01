## Módulo de Processamento

Esse é o módulo principal da biblioteca, ele concentra o uso de todas as implementações
presentes nos módulos anteriores e realiza as execuções em uma sequência definida. O objetivo
da implementação de um ProcessManager além de centralizar é possibilitar a execução dos processos
padrões de busca do melhor modelo para N modelos, sejam eles totalmente diferentes ou alterando algum detalhe.

A implementação está contida em [MultiProcessManager](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/process_manager/multi_process_manager.py#L17) que solicitará os parâmetros de
configuração que são aplicados em todos os [Pipelines](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/process_manager/pipeline.py#L9), para configurações específicas
do modelo você deve definir diretamente dentro do Pipeline de execução.

Um Pipeline contem o modelo que se deseja testar juntamente com: os parâmetros que deseja testar, 
uma implementação de busca de features, uma implementação de busca de parâmetros, um gerenciador de histórico e
um validador.

É possível definir no MultiProcessManager um único Pipeline ou uma lista deles, dependendo
da sua necessidade, além disso, você pode querer testar diferentes implementações de 
seleção de feature no mesmo modelo ou diferentes modelos com as mesmas implementações, você
é livre para escolher.