## Módulo de Histórico das Execuções

É possível ver detalhes referentes a implementação comum acessando [esse readme]().

A implementação [ScikitLearnCommonHistoryManager]() é a classe base que é utilizada para implementar
a manutenção de histórico de acordo com o objeto de resultado da validação.

No momento a única classe de histórico das implementações do scikit-learn é [ScikitLearnCrossValidationHistoryManager](), 
pois a validação cruzada foi algo que encaixou muito bem no processo, e com isso, obtemos resultados bastante robustos. 
Se futuramente surgir algum outro tipo de validação, deverá ser implementada uma classe específica pois a implementação
de manutenção do histórico precisa ter conhecimento de quais campos ela precisa manipular.
