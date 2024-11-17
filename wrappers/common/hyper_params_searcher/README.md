## Módulo Comum de Busca dos Hiperparâmetros

Esse módulo contém toda a implementação que faça sentido ser compartilhada entre as
bibliotecas utilizadas internamente. É um módulo bem simples pois as bibliotecas possuem
muitas diferenças nas suas implementações, por conta disso, não é possível declarar
muitas funcionalidades como comuns.

A implementação abstrata responsável é [CommonHyperParamsSearch](), sendo possível estendê-la
e realizar as implementações específicas da biblioteca para buscar o melhor modelo.