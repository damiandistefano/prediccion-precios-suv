# GangaSUV

Motor de predicción de precios y detección de oportunidades para el mercado de SUVs usados en Argentina. A partir de miles de publicaciones reales (marca, modelo, año, kilometraje, motor, tipo de combustible, etc.), el sistema estima el precio de mercado de un vehículo y permite identificar autos publicados por debajo de su valor esperado.

## Tech Stack

- **Lenguaje:** Python 3.10
- **Manipulación de datos:** pandas, NumPy
- **Modelado:** scikit-learn (Random Forest, `train_test_split`, `KFold`), TensorFlow / Keras (red neuronal densa)
- **Regresión lineal:** implementación propia desde cero (mínimos cuadrados vía pseudo-inversa/SVD y descenso por gradiente, con regularización L1/L2)
- **Visualización:** Matplotlib, Seaborn
- **Entorno de desarrollo:** Jupyter Notebook

## Características clave

- **Pipeline de limpieza configurable**: corrección de marcas mal tipeadas, conversión de precios a USD, cálculo de antigüedad, parseo de kilometraje y cilindrada, codificación one-hot y filtrado de outliers por IQR, todo controlable por parámetros.
- **Ingeniería de features**: precio por kilómetro, antigüedad al cuadrado, interacción cilindrada×km y frecuencia de marca/modelo.
- **Tres modelos de predicción comparados**: regresión lineal propia, Random Forest con búsqueda de hiperparámetros por validación cruzada, y una red neuronal densa entrenada con Keras.
- **Detección de gangas**: ranking de los autos con mayor diferencia entre precio publicado y precio predicho por el modelo.
- **Visualizaciones**: comparación de error entre modelos, distribución de variables y precio según tipo de combustible.

## Resultados

Comparación de los modelos evaluados sobre el conjunto de validación/test (~16.500 publicaciones tras la limpieza de outliers):

| Modelo                  | RMSE (USD) |
|--------------------------|-----------:|
| Random Forest (tuneado)   |      5.404 |
| Red Neuronal (Keras)      |      4.185 |

La red neuronal fue el modelo con mejor desempeño y se usó para generar las predicciones finales sobre el set de test.

## Demo

Precio real vs. precio predicho sobre una muestra de publicaciones que el modelo no vio durante el entrenamiento:

| Marca      | Modelo      | Precio real (USD) | Precio predicho (USD) |
|------------|-------------|-------------------:|------------------------:|
| Citroën    | C4 Cactus   |              15.900 |                   18.132 |
| Nissan     | Kicks       |              19.911 |                   20.058 |
| Volkswagen | Tiguan      |              12.149 |                   15.941 |
| Nissan     | X-Trail     |              50.000 |                   42.817 |
| Nissan     | Kicks       |              31.000 |                   29.268 |
| SsangYong  | Musso       |              10.968 |                   20.547 |
| Renault    | Duster      |              13.000 |                   18.024 |
| Jeep       | Renegade    |              19.827 |                   31.965 |

## Estructura del proyecto

```
.
├── Notebook_SUVs.ipynb        # Notebook principal con el análisis end-to-end
├── dataset/
│   ├── raw/                   # Dataset original
│   ├── processed/             # Datasets transformados / features generados
│   └── test_masked/           # Set de test para evaluación final
├── models/
│   ├── linear_regression.py   # Regresión lineal implementada desde cero
│   └── nn.py                  # Red neuronal (Keras) + validación cruzada
└── src/
    ├── data_cleaner.py        # Pipeline de limpieza principal
    ├── data_cleaner_2.py      # Variante del pipeline de limpieza
    ├── data_exploration.py    # Funciones de EDA
    ├── train_val_models.py    # Entrenamiento y evaluación de la regresión lineal
    ├── cross_val.py           # Random Forest + búsqueda de hiperparámetros por CV
    ├── extensiones_rf.py      # Detección y ranking de autos subvaluados
    ├── metrics.py             # MSE, RMSE, MAE
    └── plots.py                # Comparación de RMSE validación vs. test
```

## Instalación y ejecución

1. Cloná el repositorio y entrá al directorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd GangaSUV
```

2. Creá y activá un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalá las dependencias:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter
```

4. Levantá Jupyter y abrí el notebook principal:

```bash
jupyter notebook Notebook_SUVs.ipynb
```

El dataset de entrada se encuentra en `dataset/raw/pf_suvs_i302_1s2025.csv`. Los datasets procesados intermedios se guardan en `dataset/processed/`.
