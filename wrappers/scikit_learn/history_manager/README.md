## Módulo de Histórico das Execuções

O objetivo principal desse módulo é salvar e manipular os resultados das execuções realizadas,
possibilitando que, após uma série de tentativas de melhorar o seu modelo de machine learning,
você possa ir até um local e analisar tudo que você já tentou e redirecionar seus esforços
para um caminho diferente ou seguir no mesmo.

A implementação base é [HistoryManager](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/history_manager/README.md), esse manager é totalmente ligado a classe
de resultado da validação, ou seja, ela precisa ser implementada especificamente para 
um objeto de resultado pois os campos presentes na classe serão salvos no histórico.

A forma adotada para a persistência das informações são arquivos JSON e o modelo é salvo
com a extensão pkl, dessa forma, seria possível utilizar o JSON para obter insights montando
gráficos ou outra forma de análise e os modelos podem ser reutilizados facilmente.

Outra funcionalidade muito importante exercida por essa implementação é a capacidade de
reprodutibilidade de uma execução que foi salva anteriormente sem precisar esperar todo
o processo de seleção de features, busca de parâmetros e validação ocorrerem.
