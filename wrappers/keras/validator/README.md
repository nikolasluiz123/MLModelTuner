## Módulo de Validação das Redes Neurais

É possível ver detalhes referentes a implementação comum de validação de modelos acessando [esse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/validator/README.md).

### Validadores

Para realizar a validação das redes neurais utilizamos uma estratégia um pouco mais simples do que
a validação cruzada, ela é bem menos robusta mas, por conta das redes neurais exigirem mais processamento
foi optado por implementar esse tipo de validação mais simples e complementá-la com validações adicionais 
específicas de acordo com o objetivo da rede neural.

A implementação [KerasBasicClassifierValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/validator/basic_classifier_validator.py#L9) é responsável por realizar a validação
e retornar um objeto com os dados para os outros processos utilizarem.

Especificamente para regressão, temos [KerasBasicRegressorValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/validator/basic_regressor_validator.py#L9)

Existe também uma implementação de validação adicional chamadas [KerasAdditionalClassifierValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/validator/classifier_additional_validator.py#L8)
e [KerasAdditionalRegressorValidator](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/validator/additional_validator.py#L42) que utiliza implementações comuns para avaliar modelos desse tipo.
