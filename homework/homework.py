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


train_dataset.loc[train_dataset["EDUCATION"] > 4, "EDUCATION"] = 5  # Usar 5 para "others"
test_dataset.loc[test_dataset["EDUCATION"] > 4, "EDUCATION"] = 5



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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


# Asegúrate de que las columnas categóricas sean cadenas
categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]
for col in categorical_columns:
    x_train[col] = x_train[col].astype(int)
    x_test[col] = x_test[col].astype(int)

# Definir las categorías esperadas basadas en los datos únicos
sex_categories = sorted(x_train["SEX"].unique().tolist())
education_categories = sorted(x_train["EDUCATION"].unique().tolist()) 
marriage_categories = sorted(x_train["MARRIAGE"].unique().tolist())

print(f"SEX categories: {sex_categories}")
print(f"EDUCATION categories: {education_categories}")
print(f"MARRIAGE categories: {marriage_categories}")



# Crear el preprocesador para las variables categóricas
# OPCIÓN 1: Sin especificar categorías (más flexible)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
    ],
    remainder="passthrough"
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
import time
import time
import threading
import sys

def optimize_hyperparameters_with_progress(pipeline, x_train, y_train, param_grid):
    """
    Optimiza los hiperparámetros con barra de progreso que se actualiza en tiempo real.
    """
    
    # Calcular total de combinaciones
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    total_fits = total_combinations * 10  # 10-fold CV
    print(f"Probando {total_combinations} combinaciones con 10-fold CV = {total_fits} entrenamientos...")
    
    # GridSearchCV con verbose para capturar el progreso
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=2  # Activar verbose para seguir el progreso
    )
    
    # Variables para el progreso
    progress_info = {'current': 0, 'total': total_fits}
    
    # Función que simula el progreso basado en verbose output
    def update_progress():
        with tqdm(total=total_fits, desc="Optimizando", unit="fit") as pbar:
            start_time = time.time()
            
            # Iniciar GridSearchCV en thread separado
            def run_grid_search():
                grid_search.fit(x_train, y_train)
            
            search_thread = threading.Thread(target=run_grid_search)
            search_thread.start()
            
            # Actualizar barra cada segundo
            while search_thread.is_alive():
                time.sleep(1)
                # Estimación aproximada del progreso (no perfecta pero útil)
                elapsed = time.time() - start_time
                estimated_progress = min(int((elapsed / (total_fits * 0.5)) * total_fits), total_fits - 1)
                
                if estimated_progress > pbar.n:
                    pbar.update(estimated_progress - pbar.n)
                
                # Mostrar tiempo transcurrido
                pbar.set_postfix({"Tiempo": f"{elapsed:.0f}s"})
            
            # Esperar a que termine
            search_thread.join()
            
            # Completar barra al 100%
            pbar.update(total_fits - pbar.n)
            
            final_time = time.time() - start_time
            pbar.set_postfix({"Tiempo total": f"{final_time:.1f}s", "Score": f"{grid_search.best_score_:.4f}"})
    
    # Ejecutar con progreso
    update_progress()
    
    print(f"\n✅ Mejor score: {grid_search.best_score_:.6f}")
    print(f"🏆 Mejores parámetros: {grid_search.best_params_}")
    
    return grid_search

# Ejemplo de uso:
param_grid = {
    "classifier__n_estimators": [1500],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [15],
    "classifier__min_samples_leaf": [4],
    "classifier__max_features": ["sqrt"],
    "classifier__class_weight": ["balanced"],
    "classifier__criterion": ["entropy"],
    "classifier__bootstrap": [True],
    "classifier__random_state": [42]
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


    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Diccionarios con métricas
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, average="binary"),
        "f1_score": f1_score(y_train, y_train_pred, average="binary")
    }
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, average="binary"),
        "f1_score": f1_score(y_test, y_test_pred, average="binary")
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar en formato JSONL (una línea por dict)
    with open(output_path, "w") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")

    print(f"✅ Métricas guardadas en {output_path} (JSONL)")
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
    Calcula matrices de confusión para train/test y las guarda en formato JSONL.
    """

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a") as f:  # 👈 append para no sobrescribir las métricas
        f.write(json.dumps(train_cm_dict) + "\n")
        f.write(json.dumps(test_cm_dict) + "\n")

    print(f"✅ Matrices de confusión guardadas en {output_path} (JSONL)")


calculate_and_save_confusion_matrices(grid_search, x_train, y_train, x_test, y_test, "files/output/metrics.json")