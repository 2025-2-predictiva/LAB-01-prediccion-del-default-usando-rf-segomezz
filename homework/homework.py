# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".


import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report,precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,ParameterGrid, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from itertools import product
import gzip
import joblib
import json
import os
import pickle


# Leer los datasets descomprimidos
train_dataset = pd.read_csv("files/input/test_default_of_credit_card_clients.csv")
test_dataset = pd.read_csv("files/input/test_default_of_credit_card_clients.csv")


# --Renombrar columnas
train_dataset.rename(columns={"default payment next month": "default"}, inplace=True)
test_dataset.rename(columns={"default payment next month": "default"}, inplace=True)


# --Remueva la columna "ID".
train_dataset.drop(columns="ID",inplace=True)
test_dataset.drop(columns="ID",inplace=True)


# -- Elimine los registros con informacion no disponible.

train_dataset.dropna(inplace=True)
test_dataset.dropna(inplace=True)


# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".

# Convertir columnas (sex,education y marriage) categoricas a tipo string


train_dataset["EDUCATION"] = train_dataset["EDUCATION"].astype(str)
test_dataset["EDUCATION"] = test_dataset["EDUCATION"].astype(str)
train_dataset["EDUCATION"] = train_dataset["EDUCATION"].replace("4", "Others")
test_dataset["EDUCATION"] = test_dataset["EDUCATION"].replace("4", "Others")
print("Train categories:", train_dataset["EDUCATION"].unique())
print("Test categories:", test_dataset["EDUCATION"].unique())
train_dataset["SEX"] = train_dataset["SEX"].astype(str)
test_dataset["SEX"] = test_dataset["SEX"].astype(str)
train_dataset["MARRIAGE"] = train_dataset["MARRIAGE"].astype(str)
test_dataset["MARRIAGE"] = test_dataset["MARRIAGE"].astype(str)



#_______________________________________________________________________________________________________

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.



# Separar características (X) y variable objetivo (y)
X = train_dataset.drop(columns=["default"])
y = train_dataset["default"]

# Dividir los datos en entrenamiento (80%) y prueba (20%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#_______________________________________________________________________________________________________

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).



# Convertir todas las columnas categóricas a tipo string
categorical_columns = X.select_dtypes(include=["object", "category"]).columns


# Verifica y convierte las columnas categóricas a cadenas
for col in categorical_columns:
    x_train[col] = x_train[col].fillna("missing").astype(str)
    x_test[col] = x_test[col].fillna("missing").astype(str)

    # Verifica los valores únicos en cada columna categórica
    print(f"Valores únicos en {col} (entrenamiento): {x_train[col].unique()}")
    print(f"Valores únicos en {col} (prueba): {x_test[col].unique()}")

# Crear el preprocesador para las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"  # Dejar las columnas no categóricas sin cambios
)

# Crear el pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Ajustar el pipeline con los datos de entrenamiento
pipeline.fit(x_train, y_train)

print("Pipeline ajustado con éxito.")
#_____________________________________________________________________________________

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.


from sklearn.model_selection import GridSearchCV

def optimize_hyperparameters_with_progress(pipeline, x_train, y_train, param_grid):
    """
    Optimiza los hiperparámetros del pipeline usando GridSearchCV.
    
    Args:
        pipeline: El pipeline que contiene el modelo y preprocesamiento.
        x_train: Datos de entrenamiento (features).
        y_train: Etiquetas de entrenamiento.
        param_grid: Diccionario con los hiperparámetros a probar.
    
    Returns:
        grid_search: Objeto GridSearchCV entrenado.
    """
    # Crear el objeto GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar el modelo con los datos de entrenamiento
    grid_search.fit(x_train, y_train)
    
    return grid_search

# Ejemplo de uso:
param_grid = {
    "classifier__n_estimators": [100],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1]
}

grid_search = optimize_hyperparameters_with_progress(pipeline, x_train, y_train, param_grid)


#___________________________________________________________________________________________________________________
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.


def save_model_as_gzip(model, filepath):
    """
    Guarda un modelo como un archivo comprimido con gzip.
    
    Args:
        model: El modelo a guardar (por ejemplo, un objeto GridSearchCV).
        filepath: Ruta del archivo donde se guardará el modelo.
    """
    with gzip.open(filepath, "wb") as f:
        pickle.dump(model, f)

# Ejemplo de uso:
save_model_as_gzip(grid_search, "files/models/model.pkl.gz")
#___________________________________________________________________________________________________________________
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test, output_path):
    """
    Calcula las métricas para los conjuntos de entrenamiento y prueba y las guarda en un archivo JSONL.

    Args:
        model: Modelo entrenado.
        x_train: Datos de entrenamiento (features).
        y_train: Etiquetas de entrenamiento.
        x_test: Datos de prueba (features).
        y_test: Etiquetas de prueba.
        output_path: Ruta del archivo JSON donde se guardarán las métricas.
    """
    # Predicciones para el conjunto de entrenamiento
    y_train_pred = model.predict(x_train)
    # Predicciones para el conjunto de prueba
    y_test_pred = model.predict(x_test)

    # Calcular métricas para el conjunto de entrenamiento
    train_metrics = {
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, average="binary"),
        "f1_score": f1_score(y_train, y_train_pred, average="binary")
    }
  # Calcular métricas para el conjunto de prueba
    test_metrics = {
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, average="binary"),
        "f1_score": f1_score(y_test, y_test_pred, average="binary")
    }

    # Guardar las métricas en un archivo JSONL (una línea por objeto JSON)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Crear directorios si no existen
    with open(output_path, "w") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")


# Ejemplo de uso:
calculate_and_save_metrics(grid_search, x_train, y_train, x_test, y_test, "files/output/metrics.json")


#___________________________________________________________________________________________________________________
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}





def calculate_and_save_confusion_matrices(model, x_train, y_train, x_test, y_test, output_path):
    """
    Calcula las matrices de confusión para los conjuntos de entrenamiento y prueba y las guarda en un archivo JSON.

    Args:
        model: Modelo entrenado.
        x_train: Datos de entrenamiento (features).
        y_train: Etiquetas de entrenamiento.
        x_test: Datos de prueba (features).
        y_test: Etiquetas de prueba.
        output_path: Ruta del archivo JSON donde se guardarán las matrices de confusión.
    """
    # Predicciones para el conjunto de entrenamiento
    y_train_pred = model.predict(x_train)
    # Predicciones para el conjunto de prueba
    y_test_pred = model.predict(x_test)

    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Convertir las matrices de confusión al formato especificado
    train_cm_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }

    test_cm_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    # Crear los directorios necesarios si no existen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar las matrices de confusión en un archivo JSON
    with open(output_path, "w") as f:
        json.dump([train_cm_dict, test_cm_dict], f, indent=4)

# Ejemplo de uso:
# calculate_and_save_confusion_matrices(best_model, x_train, y_train, x_test, y_test, "files/output/metrics.json")