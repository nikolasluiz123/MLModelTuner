## Módulo de Busca de Hiperparâmetros Keras Tuner

É possível ver detalhes referentes a implementação comum de busca de parâmetros acessando [esse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/hyper_params_searcher/README.md).

A biblioteca [Keras](https://keras.io/api/) é muito expressiva na implementação de redes neurais
de qualquer tipo, sabendo disso, a biblioteca wrapper implementada utiliza algumas da implementações
de busca de parâmetros mais utilizadas, as quais serão abordadas a seguir.

Abaixo temos o exemplo de uma implementação de uma rede neural:
```
class ExampleKerasHyperModel(HyperModel):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

    def build(self, hp):
        self.base_model.trainable = False
        last_layer = self.base_model.get_layer('mixed5')

        model_extension = Flatten()(last_layer.output)

        dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=512, step=32)
        dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=512, step=32)
        dropout_rate1 = hp.Float('dropout_rate1', min_value=0.1, max_value=0.5, step=0.1)
        dropout_rate2 = hp.Float('dropout_rate2', min_value=0.1, max_value=0.5, step=0.1)

        model_extension = Dense(units=dense_units_1, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_1')(model_extension)
        model_extension = Dropout(rate=dropout_rate1)(model_extension)
        model_extension = Dense(units=dense_units_2, activation='relu')(model_extension)
        model_extension = BatchNormalization(name='batch_norm_2')(model_extension)
        model_extension = Dropout(rate=dropout_rate2)(model_extension)
        model_extension = Dense(self.num_classes, activation='softmax')(model_extension)

        model = keras.models.Model(inputs=self.base_model.input, outputs=model_extension)

        learning_rate = hp.Float(name='learning_rate', min_value=0.0001, max_value=0.01)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model
```

Basicamente a definição dos parâmetros que desejamos testar é feita através desse tipo de código:
```dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=512, step=32)``` que é capaz de
retornar um valor (nesse caso inteiro) entre o mínimo e máximo informado e em cada tentativa de obter
um novo valor será sempre com uma diferença de 32 (step).

Com isso é possível passar o valor retornado por esse código para o atributo que desejar, esse exemplo
específico serve para definir a quantidade de neurônios da primeira camada Dense dessa rede. Todas
as implementações de busca trabalham dessa mesma maneira.

### Hyperband

Essa implementação do keras é uma melhoria da busca aleatória, que se baseia na alocação adaptativa de 
recursos e parada antecipada. Em resumo, são realizados processos capazes de 'aprender' com execuções 
anteriores e também é possível parar o processo quando for notado que o modelo não está melhorando.

| Implementação Externa                                                              | Implementação Wrapper                                                                                                                             | 
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Hyperband](https://keras.io/api/keras_tuner/tuners/hyperband/#hyperband-class)    | [KerasHyperBandSearcher](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/hyper_params_search/hyper_band_searcher.py#L7) | 

### RandomSearch

Essa implementação tem como objetivo trabalhar com uma grande quantidade de possibilidades de valores,
os quais serão selecionados apenas alguns para realização do teste.

| Implementação Externa                                            | Implementação Wrapper                                                                                                                        | 
|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [RandomSearch](https://keras.io/api/keras_tuner/tuners/random/)  | [KerasRandomSearcher](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/hyper_params_search/random_searcher.py#L7)   | 

### GridSearch

Essa implementação é parecida com RandomSearch, mas ao invés de testar apenas uma parte do conjunto
de possibilidades serão testadas todas as possibilidades. Devemos ter cuidado com a gama de valores
que testamos pois o processo pode ser demorado, principalmente se for uma rede neural muito grande.

| Implementação Externa                                           | Implementação Wrapper                                                                                                                    | 
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| [GridSearch](https://keras.io/api/keras_tuner/tuners/grid/)     | [KerasGridSearcher](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/keras/hyper_params_search/grid_searcher.py#L7)   | 

