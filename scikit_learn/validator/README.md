## Módulo de Validação dos Modelos

Esse módulo tem como foco validar o modelo encontrado com os melhores parâmetros e conseguir
demonstrar se realmente o modelo trará os resultados desejados. Para isso foi implementada
a classe [BaseValidator]() a qual define aquilo que é esperado por um validador e todos
os wrappers implementados deverão utilizar essa classe como base.

### Validação Cruzada

A validação cruzada é a validação principal aplicada no processo pois é uma das mais
robustas e possibilita a obtenção de métricas matemáticas comuns entre os modelos de
classificação e regressão, isso signifca que podemos reaproveitar a implementação. Além
disso ela costuma trazer um nível maior de confiança pelo fato de realizar diversos processos
separando os dados de maneiras diferentes, veja mais detalhes sobre a validação cruzada
utilizada na [documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).

A classe wrapper para execução dessa validação cruzada no processo de busca do melhor
modelo é [CrossValidator](), essa implementação utiliza a função cross_val_score já mencionada
e ao fim do processo são realizados cálculos de métricas matemáticas, as quais são salvas
em um objeto e o mesmo é retornado para que o fluxo seja seguido. Esse objeto de retorno
será utilizado principalmente pelo módulo de histórico, o qual é detalhado [nesse readme]().

Os objetos de resultado das validações devem implementar [ValidationResult](). A classe
concreta existente para o uso das implementações atualmente é [CrossValidationResult](),
nele podemos armazenar métricas matemáticas básicas: média, mediana e desvio padrão. Além
disso, também armazenamos o modelo que foi validado e qual a métrica de avaliação utilizada
na validação.

### Validações Específicas de Classificação

Além do uso de métricas matemáticas gerais podemos utilizar algumas implementações fornecidas
pelo scikit-learn que são especificamente úteis para os cenários de classificação. A implementação
[ClassifierAdditionalValidator]() faz uso de [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
e [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) para trazer
uma visão diferenciada sobre o modelo e possibilitar o julgamento dele como uma validação final,
utilizada apenas após validá-lo de forma cruzada.

A ideia do uso dessa implementação é de fato recuperar o modelo utilizando a implementação de histórico
e utilizar os mesmos dados para realizar o fit e predict, em seguida utilizar a previsão para calcular
a confusão e obter o relatório.

