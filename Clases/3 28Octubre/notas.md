# Clase 3
## Técnicas de reducción de datos
- Un conjunto de datos es una matriz m filas x n columnas
- Si tenemos m's y n's muy grandes es probable que el computador no tenga capacidad de procesarlas
    - Para solucionarlo tenemos que reducirl el tamaño de la matriz eliminando filas o columnas
- Para eliminar la filas, m >> 10n, al menos debemos tener 10 veces más datos que columnas
    - Eliminamos haciendo un submuestreo(muestro aleatorio, etc)
    - Para eliminar filas debemos tener suficientes datos
- Para eliminar columnas:
    1. Eliminar las columnas que están mas correlacionadas
    2. Aporte información principal (Análisis de componentes principales(PCA Principal Components Analysis))

## Glosario
- `>>`: Significa mucho mayor