## Módulo de Pré-Processamento dos Dados

Esse módulo tem um objetivo bem simples e direto, basicamente fornece uma implementação
que possibilita concentrar todos os processos necessários para obter o conjunto
de dados que será utilizado no processo de treino de um modelo de machine learning.

A implementação abstrata responsável por isso é [CommonDataPreProcessor](https://github.com/nikolasluiz123/MLModelTuner/blob/master/wrappers/common/data_pre_processor/common_data_pre_processor.py#L5), sendo possível estendê-la
e realizar os processos específicos que sua base e/ou modelo necessitar.
