# Clase 2
- ¿Qué es un conjunto de datos?
  - Es un subconjunto de R^n, donde n es el número de columnas que tiene los datos
---
## Tipos de aprendizajes
- Un modelo es cualquier función que vaya desde R^n -> R^m
  - Si se los resultados de R^m es un **aprendizaje supervisado** caso contrario es un **aprendizaje no supervisado**
  - A los resultados le vamos a llamar **etiquetas o variable objetivo**
    - Las etiquetas dependen del objetivo
  - Con los mismos datos puedo emplear aprendizaje supervisado y no supervisado, lo que cambia es nuestro objetivo.
  - **Aprendizaje no supervisado:** Busca descubrir estructura en los datos sin utilizar etiquetas objetivo. En particular, el agrupamiento busca determinar una función: f: R^n ->{1, 2, ..., k}
  - Aprendizaje supervisado se ocupa para **predecir**
  - Aprendizaje no supervisado se utiliza para **agrupar**
---
## Tipos de tareas
1. **Clasificación:** Es una tarea de aprendizaje supervisado donde el conjunto de etiquetas es un conjunto finito y discreto.
2. **Regresión:** Es una tarea de aprendizaje supervisado donde el conjunto de etiquetas es un conjunto continuo
3. **Agrupamiento:** Una tarea de aprendizaje no supervisado donde se busca agrupar las instancias en función de sus características
---
## Ventana de contexto
- Los LLM funcionan prediciendo la siguiente palabra basándose en el texto que ya tienen delante.
- Lo hacen paso a paso (de forma autorregresiva): leen un bloque de palabras, predicen una nueva, la añaden al texto y repiten el ciclo.
- La ventana de contexto es el límite máximo de palabras que el modelo puede recordar y mirar al mismo tiempo durante este proceso.
- Si el modelo comete un error al predecir una palabra, ese error pasa a formar parte del texto que leerá después, lo que puede provocar que el hilo de la conversación empiece a desviarse(propagación de errores).
---
## Modelos principales
| Método | Clasificación | Regresión | Agrupamiento |
| --- | --- | --- | --- |
| k-NN | × | × |  |
| SVM | × | × |  |
| Árboles de decisión | × | × |  |
| Redes neuronales | × | × |  |
| Agrupamiento jerárquico |  |  | × |
| k-means |  |  | × |
| DBSCAN |  |  | × |
---
## Métricas
- ¿Cómo medir la distancia?
- La distancia se puede medir en cualquier conjunto
- Es una función que toma un par de elementos y me da como resultado un número real
- Para llamarla distancia tiene que ser no negativa, es decir mayor a 0, tiene que preservar la identidad, tiene que ser simétrica(la distancia de x a y es igual a la distancia de y a x) y la propiedad de desigualdad triangular
- `d ∶ X × X → ℝ`
- X es cualquier conjunto de elementos (pueden ser puntos en un plano, números, etc.).
- X * X (el producto cartesiano) significa que la función toma una pareja de elementos del conjunto, por ejemplo, x e y.
- → ℝ significa que el resultado de medir esa pareja es un número real (un valor numérico, como 5, 2.3 o 0).
- d(x, y) es la regla o función que calcula la distancia entre el elemento x y el elemento y.
---
## Glosario
- **Cardinalidad:** La cantidad de elementos dentro de un conjunto de datos, se representa [|A| = x]; donde x es la cantidad de elementos
- En base de datos relacionales:
  - Instancia son las filas
  - Atributos las columnas
  - Y un un dato en específico es un valor
- **Conjunto continuo:** Es una colección de valores donde, entre cualquier par de elementos, siempre existe una infinidad de otros elementos intermedios.
---
## Materiales usados en clase
- [Resumen material de clase del profesor](https://andres-merino.github.io/AprendizajeAutomaticoInicial-05-N0105/2-Resumenes/Resumen02.pdf)
- [Material de clase](https://andres-merino.github.io/AprendizajeAutomaticoInicial-05-N0105/2-Clases/Clase02.pdf)