# Predicción de Precios de SUVs 🚗

Pipeline de Machine Learning en Python para predecir el precio de venta de SUVs de segunda mano en Argentina a partir de datos de publicaciones (marca, modelo, año, kilometraje, motor, tipo de combustible, etc.), y para detectar autos **subvaluados** (publicados por debajo de su precio de mercado estimado). Desarrollado como Proyecto Final para la materia de Aprendizaje Automático (UDESA).

## 🛠️ Tech Stack

- **Lenguaje:** Python 3.10
- **Manipulación de datos:** pandas, NumPy
- **Modelado:** scikit-learn (Random Forest, `train_test_split`, `KFold`), TensorFlow / Keras (red neuronal densa)
- **Regresión lineal:** implementación propia desde cero (mínimos cuadrados vía pseudo-inversa/SVD y descenso por gradiente, con regularización L1/L2)
- **Visualización:** Matplotlib, Seaborn
- **Entorno de desarrollo:** Jupyter Notebook

## ✨ Funcionalidades principales

- **Limpieza y preprocesamiento de datos** (`src/data_cleaner.py`, `src/data_cleaner_2.py`): corrección de marcas mal tipeadas, conversión de precios a USD, cálculo de antigüedad, parseo de kilometraje y cilindrada, codificación one-hot, agrupación de categorías poco frecuentes y filtrado de outliers por IQR.
- **Ingeniería de features**: variables derivadas como precio por kilómetro, antigüedad al cuadrado, interacción cilindrada×km y frecuencia de marca/modelo (activables por configuración).
- **Tres enfoques de modelado**:
  - Regresión Lineal implementada a mano (`models/linear_regression.py`), con entrenamiento por pseudo-inversa o gradiente descendente y regularización L1/L2.
  - Random Forest con búsqueda de hiperparámetros por validación cruzada (`src/extensiones_rf.py`).
  - Red Neuronal (Keras) con validación cruzada y early stopping (`models/nn.py`).
- **Evaluación de modelos** con métricas MSE, RMSE y MAE (`src/metrics.py`).
- **Visualizaciones**: comparación de RMSE entre validación/test, distribución de variables, boxplots por tipo de combustible.
- **Detección de oportunidades**: identificación y ranking de los autos más subvaluados respecto a la predicción del modelo (`src/plots.py`).

## 📂 Estructura del proyecto

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
    ├── data_cleaner_2.py      # Variante del pipeline de limpieza
    ├── data_exploration.py    # Funciones de EDA
    ├── train_val_models.py    # Entrenamiento y evaluación de la regresión lineal
    ├── cross_val.py           # Utilidades de validación cruzada
    ├── extensiones_rf.py      # Random Forest + grid search
    ├── metrics.py             # MSE, RMSE, MAE
    └── plots.py                # Visualizaciones de resultados
```

## 🚀 Instalación y ejecución (Linux / Terminal)

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

> El dataset de entrada se encuentra en `dataset/raw/pf_suvs_i302_1s2025.csv`. Los datasets procesados intermedios se guardan en `dataset/processed/`.

## 📸 Demo

*(Agregar aquí un GIF o captura de pantalla del notebook en ejecución, por ejemplo el gráfico de comparación de RMSE o el ranking de autos subvaluados: `docs/demo.gif` o `docs/screenshot.png`)*

---

## 2. Descripción para GitHub (campo "About")

> Pipeline de Machine Learning en Python (Random Forest, red neuronal y regresión lineal propia) para predecir precios de SUVs usados y detectar autos subvaluados.

---

## 3. Code Review Express

Antes de subir el repo al portafolio, conviene resolver lo siguiente:

**🐛 Bugs**
- **`src/train_val_models.py`**: usa `np.float64` y `np.column_stack` (vía `LinearReg`) sin importar `numpy` en el archivo (`import pandas as pd` está, pero falta `import numpy as np`). Va a explotar con `NameError` en `prepare_data`.
- **`src/train_val_models.py` → `run_experiment`**: llama a `train_pred_linear_reg(processor, X_train_norm, X_val_norm, y_train, y_val.values, ...)` con `...` (Ellipsis) literal como argumento posicional. Eso pisa el parámetro `metodo` con `Ellipsis` en vez de `"pinv"`, así que nunca entra a las ramas `if metodo == "pinv"` / `elif metodo == "gd"` y el modelo queda con pesos en cero (sin entrenar). Parece código incompleto que quedó a medio escribir.
- **`models/nn.py`**: `cross_validate_nn` usa `DataProcessor` (de `data_cleaner.py`) mientras que el resto del proyecto (notebook) parece usar `DataProcessor2` en paralelo — dos clases casi idénticas mantenidas por separado es fuente de bugs por desincronización (ver duplicación abajo).

**🧹 Malas prácticas / mejoras rápidas**
- **Duplicación de código**: `src/data_cleaner.py` (`DataProcessor`) y `src/data_cleaner_2.py` (`DataProcessor2`) son casi idénticas. Conviene unificarlas en una sola clase parametrizable para evitar mantener dos pipelines de limpieza en paralelo.
- **Código muerto**: `models/nn.py` tiene ~65 líneas comentadas (una versión vieja de `cross_validate_nn`, líneas 74-138). Si ya no se usa, eliminarlas en vez de dejarlas comentadas.
- **`import *`**: `src/train_val_models.py` hace `from src.metrics import*` (wildcard import, además sin espacio). Preferible importar explícitamente (`from src.metrics import mse, rmse, mae`).
- **Imports fuera de lugar**: en `src/extensiones_rf.py`, `from sklearn.model_selection import train_test_split` está en medio del archivo (línea 72) en lugar de al principio.
- **Estado global mutable**: `param_grid` en `src/extensiones_rf.py` está definido como variable a nivel de módulo (línea 141) — mejor pasarlo como argumento o definirlo en el notebook/config.
- **Magic numbers**: la tasa de conversión USD (`1185.26`) y el año de referencia (`2025 - df["Año"]`) están hardcodeados dentro del pipeline de limpieza. Convendría pasarlos por `config` para que el pipeline no quede atado a una fecha/tipo de cambio fijos.
- **Archivos que no deberían estar en git**: `.DS_Store`, `dataset/.DS_Store`, `models/__pycache__/` y `src/__pycache__/` están versionados. Agregar un `.gitignore` (con `__pycache__/`, `*.pyc`, `.DS_Store`, `venv/`) y sacarlos del repo con `git rm -r --cached`.
- **Falta `requirements.txt`**: no hay archivo de dependencias; para que el proyecto sea reproducible por un tercero conviene agregar uno (`pip freeze > requirements.txt` o una lista curada).
- **Reporte por `print`**: funciones como `cross_validate_rf` y el filtrado de outliers usan `print()` para reportar progreso/resultados. Está bien para un notebook exploratorio, pero si se reutiliza como librería conviene loguear con el módulo `logging`.
