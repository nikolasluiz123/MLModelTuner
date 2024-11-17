## Módulo de Gerenciamento de Multi Processos

É possível ver detalhes referentes a implementação comum dos gerenciadores acessando [esse readme](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/process_manager/README.md).

A implementação específica do scikit-learn é [ScikitLearnMultiProcessManager](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/scikit_learn/process_manager/multi_process_manager.py#L15) que pode ser utilizado para avaliar
quaisquer modelos da biblioteca.

O pipeline é definido por [ScikitLearnPipeline](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/scikit_learn/process_manager/pipeline.py#L14) pois temos classes específicas para cada um dos processos realizados
no ProcessManager.