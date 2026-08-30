# Predicción de Precios de SUVs

Pipeline de Machine Learning en Python para predecir el precio de venta de SUVs de segunda mano en Argentina a partir de datos de publicaciones (marca, modelo, año, kilometraje, motor, tipo de combustible, etc.), y para detectar autos subvaluados (publicados por debajo de su precio de mercado estimado). Desarrollado como Proyecto Final para la materia de Aprendizaje Automático (UDESA).

## Tech Stack

- **Lenguaje:** Python 3.10
- **Manipulación de datos:** pandas, NumPy
- **Modelado:** scikit-learn (Random Forest, `train_test_split`, `KFold`), TensorFlow / Keras (red neuronal densa)
- **Regresión lineal:** implementación propia desde cero (mínimos cuadrados vía pseudo-inversa/SVD y descenso por gradiente, con regularización L1/L2)
- **Visualización:** Matplotlib, Seaborn
- **Entorno de desarrollo:** Jupyter Notebook

## Funcionalidades principales

- **Limpieza y preprocesamiento de datos** (`src/data_cleaner.py`, `src/data_cleaner_2.py`): corrección de marcas mal tipeadas, conversión de precios a USD, cálculo de antigüedad, parseo de kilometraje y cilindrada, codificación one-hot, agrupación de categorías poco frecuentes y filtrado de outliers por IQR.
- **Ingeniería de features**: variables derivadas como precio por kilómetro, antigüedad al cuadrado, interacción cilindrada×km y frecuencia de marca/modelo (activables por configuración).
- **Tres enfoques de modelado**:
  - Regresión Lineal implementada a mano (`models/linear_regression.py`), con entrenamiento por pseudo-inversa o gradiente descendente y regularización L1/L2.
  - Random Forest con búsqueda de hiperparámetros por validación cruzada (`src/cross_val.py`).
  - Red Neuronal (Keras) con validación cruzada y early stopping (`models/nn.py`).
- **Evaluación de modelos** con métricas MSE, RMSE y MAE (`src/metrics.py`).
- **Visualizaciones**: comparación de RMSE entre validación/test (`src/plots.py`), distribución de variables y precio por tipo de combustible (`src/data_exploration.py`).
- **Detección de oportunidades**: identificación y ranking de los autos más subvaluados respecto a la predicción del modelo (`src/extensiones_rf.py`).

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
    ├── data_cleaner.py        # Pipeline de limpieza principal (DataProcessor)
    ├── data_cleaner_2.py      # Variante del pipeline de limpieza (DataProcessor2)
    ├── data_exploration.py    # Funciones de EDA
    ├── train_val_models.py    # Entrenamiento y evaluación de la regresión lineal
    ├── cross_val.py           # Random Forest + búsqueda de hiperparámetros por CV
    ├── extensiones_rf.py      # Detección y ranking de autos subvaluados
    ├── metrics.py             # MSE, RMSE, MAE
    └── plots.py                # Comparación de RMSE validación vs. test
```

## Instalación y ejecución (Linux / Terminal)

1. Cloná el repositorio y entrá al directorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd "Proyecto Final"
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

## Demo

Resultado real generado por el modelo sobre el set de test (`dataset/test_masked/`), combinando los datos de entrada (`SUVS_2025-test-masked.csv`) con las predicciones de precio del modelo (`SUVS_2025-test-masked-predictions.csv`):

| Marca      | Modelo         | Año  | Kilómetros | Precio predicho (USD) |
|------------|----------------|------|-----------:|-----------------------:|
| Kia        | Soul           | 2015 |    107.000 |                  16.711 |
| Jeep       | Compass        | 2025 |      4.000 |                  39.017 |
| Volkswagen | T-Cross        | 2025 |          0 |                  29.366 |
| Ford       | Territory      | 2024 |     13.000 |                  38.243 |
| Jeep       | Compass        | 2024 |     25.200 |                  36.263 |
| Chery      | Tiggo          | 2013 |    116.000 |                  10.664 |
| Chevrolet  | Tracker        | 2016 |     98.300 |                  16.384 |
| Toyota     | Corolla Cross  | 2025 |          0 |                  36.668 |

*(Agregar aquí también un GIF o captura del notebook en ejecución, por ejemplo el gráfico de comparación de RMSE generado por `src/plots.py`: `docs/demo.gif` o `docs/screenshot.png`)*

## Descripción para GitHub (campo "About")

> Pipeline de Machine Learning en Python (Random Forest, red neuronal y regresión lineal propia) para predecir precios de SUVs usados y detectar autos subvaluados.

## Code Review Express

Estado de los puntos detectados en la revisión de código:

**Bugs corregidos**
- `src/train_val_models.py`: faltaba `import numpy as np` (el archivo usaba `np.float64`/`np.column_stack` sin importarlo). Corregido.
- `src/train_val_models.py` → `run_experiment`: se pasaba un `...` (Ellipsis) literal como argumento posicional, pisando el parámetro `metodo` y dejando el modelo sin entrenar (pesos en cero). Corregido: ahora se pasan `metodo`, `reg` y `l2` explícitamente.

**Limpieza aplicada**
- `models/nn.py`: se eliminó un bloque de ~65 líneas de código muerto (versión vieja y comentada de `cross_validate_nn`).
- `src/train_val_models.py`: se reemplazó el `from src.metrics import*` (wildcard import) por imports explícitos (`mse`, `rmse`, `mae`).
- `src/cross_val.py`: se movió el `from sklearn.model_selection import train_test_split` (estaba en medio del archivo) junto al resto de los imports al inicio.
- `src/cross_val.py`: el `param_grid` de Random Forest, definido como variable global a nivel de módulo, se renombró a `DEFAULT_PARAM_GRID` y ahora `evaluate_datasets` lo recibe como parámetro opcional en lugar de depender de estado global.
- `src/data_cleaner.py` y `src/data_cleaner_2.py`: la tasa de conversión a USD (`1185.26`) y el año de referencia para calcular antigüedad (`2025`) estaban hardcodeados. Ahora son parámetros de `config` (`usd_conversion_rate`, `reference_year`), con el mismo valor por defecto para no alterar el comportamiento actual.
- Se agregó `.gitignore` (`__pycache__/`, `*.pyc`, `.DS_Store`, `venv/`, `.ipynb_checkpoints/`) y se sacaron del tracking de git los archivos `.DS_Store` y `__pycache__/` que estaban versionados por error.

**Pendiente (a evaluar antes de subir a portafolio)**
- Duplicación de código: `src/data_cleaner.py` (`DataProcessor`) y `src/data_cleaner_2.py` (`DataProcessor2`) tienen lógica de limpieza casi idéntica pero con APIs distintas (`preprocess_global`/`preprocess_split` vs. `preprocess()` único) y ambas se usan activamente en el notebook. Unificarlas requiere actualizar todas las celdas que las instancian, así que se dejó fuera de este pase para no romper el trabajo en curso sobre el notebook.
- No hay archivo `requirements.txt`/entorno reproducible; se recomienda agregarlo (`pip freeze > requirements.txt` o una lista curada) para que el proyecto sea reproducible por un tercero.
- Uso de `print()` para reportar progreso/resultados (por ejemplo en `cross_validate_rf`). Es razonable en un notebook exploratorio, pero si el código se reutiliza como librería conviene migrar a `logging`.
