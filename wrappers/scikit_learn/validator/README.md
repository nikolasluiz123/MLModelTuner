## Módulo de Validação dos Modelos

É possível ver detalhes referentes a implementação comum acessando [esse readme]().

### Validação Cruzada

A validação cruzada é a validação principal aplicada no processo, pois é uma das mais
robustas e possibilita a obtenção de métricas matemáticas comuns entre os modelos de
classificação e regressão, isso significa que podemos reaproveitar a implementação. Além
disso, ela costuma trazer um nível maior de confiança pelo fato de realizar diversos processos
separando os dados de maneiras diferentes, veja mais detalhes sobre a validação cruzada
utilizada na [documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).

A classe wrapper para execução dessa validação cruzada no processo de busca do melhor
modelo é [ScikitLearnCrossValidator](), essa implementação utiliza a função cross_val_score já mencionada
e, ao fim do processo, são realizados cálculos de métricas matemáticas, as quais são salvas
em um objeto e o mesmo é retornado para que o fluxo seja seguido. Esse objeto de retorno
será utilizado principalmente pelo módulo de histórico, o qual é detalhado [nesse readme](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/history_manager/README.md).

A classe existente para o uso das implementações de validação cruzada atualmente é [ScikitLearnCrossValidationResult](https://github.com/nikolasluiz123/MLModelTunner/blob/master/scikit_learn/validator/results/cross_validation.py#L6),
nele podemos armazenar métricas matemáticas variadas. Além disso, também armazenamos o modelo que foi validado e qual
a métrica de avaliação utilizada na validação.

### Validações Específicas de Classificação

Além do uso de métricas matemáticas gerais podemos utilizar algumas implementações fornecidas
pelo scikit-learn que são especificamente úteis para os cenários de classificação. A implementação
[ScikitLearnClassifierAdditionalValidator](ClassifierAdditionalValidator) pode ser utilizada nesses cenários.
