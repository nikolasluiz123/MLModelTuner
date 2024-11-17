## Módulo de Gerenciamento de Histórico

Esse é um módulo muito importante da biblioteca, ele é responsável por podermos rever dados das
execuções realizadas anteriormente e comparar elas, principalmente quando estamos explorando
um modelo com valores que não sabemos se vai melhorar ou piorar nossos resultados.

Por padrão toda implementação de [CommonHistoryManager]() vai trabalhar com a manutenção dos dados
das execuções em formato de lista JSON e vai obrigatoriamente salvar os modelos obtidos para que
possam ser reutilizados. O objetivo é concentrar todas as operações com arquivos nessa implementação.