## Revisión procesado de datos
```python
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
df = pd.concat([X, y], axis=1)
df.head()

# Normalización por el máximo
x_max = X.max()
X_max_normalized = X / x_max
print("Normalización por el máximo:\n")
display(X_max_normalized.head())

# Normalización Min-Max
min_max_scaler = MinMaxScaler()
X_min_max_normalized = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
print("Normalización Min-Max:\n")
display(X_min_max_normalized.head())

# Estandarización
standard_scaler = StandardScaler()
X_standardized = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)
print("Estandarización:\n")
display(X_standardized.head())

# Igual amplitud
kbins_amplitude = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned_amplitude = pd.DataFrame(kbins_amplitude.fit_transform(X[['sepal length (cm)']]), columns=['sepal length (cm) binned'])
print("Los rangos de los bins (igual amplitud):", kbins_amplitude.bin_edges_[0])
print("Cantidad de muestras por bin (igual amplitud):", pd.Series(X_binned_amplitude['sepal length (cm) binned']).value_counts().sort_index().values, "\n")

# Igual frecuencia
kbins_frequency = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_binned_frequency = pd.DataFrame(kbins_frequency.fit_transform(X[['sepal length (cm)']]), columns=['sepal length (cm) binned'])
print("Los rangos de los bins (igual frecuencia):", kbins_frequency.bin_edges_[0])
print("Cantidad de muestras por bin (igual frecuencia):", pd.Series(X_binned_frequency['sepal length (cm) binned']).value_counts().sort_index().values)

# Característica nominal de ejemplo, independiente de la variable objetivo.
colores = pd.DataFrame({"color": ["rojo", "verde", "azul", "verde", "rojo"]})
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
colores_codificados = pd.DataFrame(
    onehot_encoder.fit_transform(colores),
    columns=onehot_encoder.get_feature_names_out(["color"])
)
display(colores_codificados)

# Varianza explicada por cada componente principal
pca = PCA()
pca.fit(X_standardized)
explained_variance = pca.explained_variance_ratio_
print("Varianza explicada por cada componente principal:\n", explained_variance)

# Grafico de la varianza explicada acumulada
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Varianza Explicada Acumulada por Componentes Principales')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid()
plt.show()

# Usar las features escaladas por StandardScaler para PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("Componentes principales (PCA):\n")
display(X_pca_df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_test_scaled = scaler_train.transform(X_test)
print("Media aprendida de entrenamiento:", scaler_train.mean_)
print("Dimensiones:", X_train_scaled.shape, X_test_scaled.shape)

```
## Funciones nuevas
- `pd.DataFrame(data, index, columns, dtype): `Convierte los datos en una tabla estructurada de dos dimensiones     
  - data: Los datos que vamos a introducir
  - index(opcional): Las etiquetas de las filas, si no la incluimos pandas le dará un valor de 0 de automáticamente
  - columns(opcional): las etiquetas o nombres de las columnas
  - dtype(opcional): Fuerza a que los datos sean de un tipo específico (int, string, float, ...) 
- `pd.Series(data, index, dytpe, name): `Convierta una lista unidimensional en una única columna de datos pandas
  - data: los datos que vamos a introducir
  - index(opcional): Las etiquetas de cada elemento
  - dtype(opcional): tipo de dato específico
  - nombre que le vamos a dar a la columna
- `pd.concat(objs, axis=0, join="outer", ignore_index=false): `concatena los datos. Toma la tabla(x) y la tabla (y) y las pega lado a lado 
  - objs: Una lista o tupla con los objetos que quieres unir (ej. [X, y]).
  - axis: Define la dirección de la unión
    - `axis=0: `Une verticalmente
    - `axis=1: `Une horizontalmente
  - join: Define cómo manejar los índices que no coinciden. 'outer' (por defecto) mantiene todo y pone valores nulos (NaN) donde falten datos, 'inner' mantiene solo las filas/columnas que ambos objetos comparten.
  - ignore_index: Si es True, borra las etiquetas originales de los índices y crea unos nuevos del 0 en adelante. Muy útil cuando unes filas y no quieres que los números de índice se repitan.
  - `MinMaxScaler: `Pertenece a la librería Scikit-learn es una herramienta de procesamiento que escala los vectores a valores numéricos de las columnas para que queden dentro de un rango específico, por defecto entre 0 y 1.
## Jupyter Notebook
- [Enlace deber](https://colab.research.google.com/github/andres-merino/AprendizajeAutomaticoInicial-05-N0105/blob/main/2-Notebooks/03-1-Procesado-Datos.ipynb#scrollTo=Q1EyfM-jFfeH)