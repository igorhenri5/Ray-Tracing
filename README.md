# Ray-Tracing

O programa foi implementado na linguagem python.
Para executá-lo, basta usar o comando:
    python tracer.py 'arquivo de entrada' 'arquivo de saída' 

Ou opcionalmente:
    python tracer.py 'arquivo de entrada' 'arquivo de saída' 'width' 'height'

O programa em questão implementa as funcionalidades básicas estabelecidas (80%), exceto pelo pigmento textmap.
Todas as operações com polyhedron foram implementadas, mas o tracing neles está desativado devido a um bug que não consegui solucionar, provavelmente de alguma peculiaridade da linguagem Python.

A implementação foi feita em uma série de funções, das seguintes etapas:
- parse
- disparo de raytrace + handling do arquivo de saída
- raytracing
    - interseção
        - interseção para Sphere
        - interseção para Polyhedron
Além de uma classe (vec3) para representar os vetores com facilidade.

No mais, a implementação consiste em efetuar os cálculos para obter a cor de cada pixel, baseados nos parametros obtidos no parsing.
Foi estruturado todo o esquema de luzes, com as 3 componentes, atenuação, refração dentro de objetos, dinâmica de sombreamento, etc.

