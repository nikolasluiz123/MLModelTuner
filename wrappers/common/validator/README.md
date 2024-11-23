## Módulo de Validação dos Modelos

Nesse módulo são concentradas as implementações comuns de validação dos modelos de machine
learning. Existem dois tipos de validadores, aqueles que utilizam processos fornecidos
pela biblioteca específica e aqueles que utilizam implementações mais gerais para avaliação
de qualquer modelo.

Os validadores que utilizam implementações específicas das bibliotecas devem estender
[CommonValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/validator/common_validator.py#L4), que contém as implementações comuns necessárias.

Os validadores adicionais que complementam a validação realizada devem estender a classe
referente ao objetivo específico do modelo, por exemplo, a classificação possui [CommonClassifierAdditionalValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/validator/common_additional_validator.py#L19)
como implementação base e para regressão temos [CommonRegressorAdditionalValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/validator/common_additional_validator.py#L115).

Essa implementação específica para validação adicional de classificadores faz uso de [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
e [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) para trazer uma visão diferenciada sobre o modelo e possibilitar o julgamento dele como uma 
validação final.

### Classe de Resultado

Para as implementações que utilizam **CommonValidator** detalhado acima, normalmente temos
a necessidade de armazenar resultados dessa validação em um objeto para que seja possível
realizar a exibição disso no console e salvar no histórico de execuções. Para resolver isso,
foi implementada a classe [CommonValidationResult](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/validator/results/common_validation_result.py#L5), dessa forma é possível garantir comportamentos
obrigatórios nas diferentes classes de resultado de validação.

Um detalhe importante é que normalmente todos os campos armazenados nas implementações específicas
por conta das diferenças entre os resultados obtidos em cada biblioteca.
