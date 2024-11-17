## Módulo de Validação das Redes Neurais

É possível ver detalhes referentes a implementação comum de validação de modelos acessando [esse readme]().

### Validadores

Para realizar a validação das redes neurais utilizamos uma estratégia um pouco mais simples do que
a validação cruzada, ela é bem menos robusta mas, por conta das redes neurais exigirem mais processamento
foi optado por implementar esse tipo de validação mais simples e complementá-la com validações adicionais 
específicas de acordo com o objetivo da rede neural.

Atualmente apenas foi implementado um validador capaz de validar redes neurais de classificação,
isso foi segregado em implementações diferentes devido a forma como os resultados do treino da rede
são retornados. A implementação [KerasBasicClassifierValidator]() é responsável por realizar esse processo
e retornar um objeto com os dados para os outros processos utilizarem.
