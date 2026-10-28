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
---
## Proceso de análisis de datos

### 1. Entendimiento del negocio

* **¿Dónde estoy trabajando?** Contexto de la industria, el dominio y los procesos actuales.
* **¿Qué quiere el negocio?** El objetivo comercial (ej. reducir fuga de clientes, predecir ventas, detectar fraudes).
* **¿Cuáles son las acciones concretas?** Qué decisiones operativas o estratégicas se tomarán con los resultados del modelo.
* **Criterio de éxito:** Métrica de negocio para medir el valor generado (ej. reducir costos un 15% o aumentar retención un 5%).

### 2. Entendimiento de los datos

* **Estructura y volumen:**
* ¿Cuántos registros (filas) y variables (columnas) tengo?
* Tipos de datos: numéricos, categóricos, texto, fechas, booleanos.

* **Diccionario de datos:** Significado preciso y origen de cada variable.
* **Calidad inicial de datos:** Identificar a simple vista datos faltantes (nulos), duplicados o inconsistencias evidentes.

### 3. Limpieza y Análisis Exploratorio de Datos (EDA)

* **Estadística descriptiva:** Medidas de tendencia central (media, mediana, moda) y dispersión (varianza, desviación estándar, rangos, percentiles).
* **Análisis de distribución y relaciones:**
* Histogramas, diagramas de caja (boxplots) para ver asimetrías y valores atípicos (*outliers*).
* Matriz de correlación para identificar variables fuertemente ligadas entre sí o con la variable objetivo.

* **Tratamiento de anomalías:**
* Manejo de valores nulos (imputación por media/mediana o eliminación).
* Decisión sobre qué hacer con los *outliers* (eliminación, transformación o conservación).

### 4. Preprocesamiento e Ingeniería de Características

* **Transformación de datos:**
* *Escalado:* Normalización (MinMaxScaler) o Estandarización (StandardScaler) para que variables con rangos grandes no dominen a las pequeñas.
* *Codificación categórica:* Convertir texto a números mediante One-Hot Encoding, Label Encoding o Target Encoding.
* *Discretización:* Agrupar variables continuas en rangos o intervalos (ej. convertir "Edad" en grupos de "Joven", "Adulto", "Mayor").


* **Ingeniería de variables (Feature Engineering):** Crear nuevas variables derivadas que aporten más información relevante.
* **Reducción de dimensionalidad y selección de variables:**
* Filtrar variables redundantes o poco informativas (Feature Selection).
* Técnicas de reducción como PCA (Análisis de Componentes Principales) o submuestreo.

### 5. Separación de datos

* **División del dataset:** Separar los datos para evitar el sobreajuste (*overfitting*) y garantizar evaluarlo con datos no vistos:
* *Conjunto de Entrenamiento (Train):* ~70-80% de los datos para enseñar al modelo.
* *Conjunto de Validación (Validation):* ~10-15% para ajustar parámetros y comparar modelos.
* *Conjunto de Prueba (Test):* ~10-15% datos completamente aislados para la evaluación final.
* **Estrategia de división:**
* *Aleatoria:* Para datos balanceados e independientes.
* *Estratificada:* Para mantener la proporción de clases en problemas desbalanceados (ej. detección de fraude donde solo el 1% es positivo).
* *Temporal:* Para series de tiempo (entrenar con el pasado, probar con el futuro).
* 20% de los datos para el entrenamiento
* 80% para la prueba
* Mientras más datos tenga para el entrenamiento mejor puede ser el modelo
* El problema es que el modelo puede momorizar los datos y no vamos a tener suficientes para probarlos
### 6. Entrenamiento del modelo

* **Selección de algoritmos:** Probar distintos tipos de modelos candidatos según el problema (Regresión, Árboles de Decisión, Random Forest, XGBoost, Redes Neuronales, etc.).
* **Ajuste de hiperparámetros:** Optimizar la configuración interna de los modelos (GridSearch, RandomSearch u optimización bayesiana).
* **Manejo del desbalance:** Aplicar técnicas como sobremuestreo (SMOTE) o ajuste de pesos si una clase es muy minoritaria.

### 7. Evaluación del modelo 

* **Métricas de rendimiento:**
* *Clasificación:* Matriz de confusión, Accuracy, Precision, Recall, F1-Score, ROC-AUC.
* *Regresión:* RMSE, MAE, R², MSE.
* **Detección de problemas:** Verificar que no exista sobreajuste (*overfitting*) ni subajuste (*underfitting*).
* **Interpretabilidad:** Analizar la importancia de las variables (Feature Importance, SHAP values) para explicar cómo decide el modelo.
---
## Pandas
- Manejo de conjunto de datos
- Puede leer csv(comma-seperated values)
    - Es una forma económica de guardar datos, no tiene formato(plain text)
    - Si está localmente debe estar en la misma carpeta en la que estamos compilando
```python
# importamos las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # librería para generar gráficos

# Cargar el conjunto de datos
url = 'https://raw.githubusercontent.com/mGalarnyk/Tutorial_Data/master/King_County/kingCountyHouseData.csv' # cargamos el conjunto de datos en línea
df = pd.read_csv(url) # hacemos que panda lea los datos
# Seleccionar las columnas de interés, definimos los atributos de interés
columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','price']
df = df.loc[:, columns]
df.head(10)

# Dividir el conjunto de datos en características y etiquetas
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
X = df.loc[:, features]
y = df.loc[:, ['price']]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #test_size = 0.2 -> le damos el 20% de los datos para el training del modelo

# Reviso las dimensiones de los conjuntos de datos
print('Dimensiones de los conjuntos de datos:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# Grafico la distribución del precio en el conjunto de entrenamiento y prueba
plt.figure(figsize=(10, 4))

# Histograma para y_train
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30, alpha=0.7, edgecolor='black', label='Train')
plt.title('Distribución de y en Train')
plt.xlabel('y')
plt.ylabel('Frecuencia')
plt.legend()

# Histograma para y_test
plt.subplot(1, 2, 2)
plt.hist(y_test, bins=30, alpha=0.7, edgecolor='black', label='Test', color='orange')
plt.title('Distribución de y en Test')
plt.xlabel('y')
plt.ylabel('Frecuencia')
plt.legend()

plt.tight_layout()
plt.show()

```
- `stratify:` trata de que la distrubución sea la misma tanto en el training como en el test
---
## Glosario
- `>>`: Significa mucho mayor

## Materiales de clase: 
- [Resumen de la clase](https://andres-merino.github.io/AprendizajeAutomaticoInicial-05-N0105/2-Resumenes/Resumen03.pdf)
- [Materiales Clase 03](https://andres-merino.github.io/AprendizajeAutomaticoInicial-05-N0105/2-Clases/Clase03.pdf)