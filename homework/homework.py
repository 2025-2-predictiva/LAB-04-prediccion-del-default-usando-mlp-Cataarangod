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
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
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
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import gzip
import json
import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, index_col=False, compression="zip")
    df_test = pd.read_csv(test_path, index_col=False, compression="zip")

    print("Datos cargados exitosamente")

    return df_train, df_test


def preprocess_data(df, set_name):
    # Renombrar columna
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con informacion no disponible
    df = df.dropna()

    # Eliminar valores = 0
    # y agrupar valores de EDUCATION > 4
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

    print(f"Preprocesamiento de datos {set_name} completado")

    return df


def split_features_target(df, target_name):
    X = df.drop(columns=[target_name])
    y = df[target_name]

    print("División de características y target completada")

    return X, y


def pipeline_definition(df):
    # Definir las variables categóricas y numéricas
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]
    numerical_features = df.columns.difference(categorical_features).tolist()

    print("Variables categóricas: " + str(categorical_features))
    print("Variables numéricas: " + str(numerical_features))

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    # Crear el pipeline
    # 1) one-hot-encoding + numeric scaling (preprocessor)
    # 2) PCA (todas las componentes)
    # 3) Selección de K mejores
    # 4) Red neuronal de tipo MLP
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
            ("pca", PCA()),
            ("classifier", MLPClassifier(max_iter=15000)),
        ]
    )

    print("Definición del pipeline completada")

    return pipeline


def hyperparameter_optimization(pipeline, X_train, y_train):
    param_grid = {
        "pca__n_components": [None, 20, 21],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60), (100, 50, 30, 20)],
        "classifier__alpha": [0.26, 0.30],
        "classifier__learning_rate_init": [0.001, 0.01],
        "classifier__learning_rate": ["adaptive", "constant"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        verbose=1,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    print("Optimización de hiperparámetros completada")

    return grid_search


def save_model(model, model_path):
    # Guardar el modelo comprimido con gzip
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with gzip.open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Modelo guardado exitosamente")


def calculate_metrics(model, X, y, dataset_type):
    y_pred = model.predict(X)

    # Calcular métricas
    precision = precision_score(y, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision,
        "balanced_accuracy": balanced_acc,
        "recall": recall,
        "f1_score": f1,
    }

    # Calcular matriz de confusión
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    print(f"Cálculo de métricas completado para el conjunto de {dataset_type}")

    return metrics, cm_dict


def save_metrics(metrics, metrics_path):
    # Guardar las métricas en un archivo json
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

    print("Métricas guardadas exitosamente")


def main():
    df_train, df_test = load_data(
        "files/input/train_data.csv.zip", "files/input/test_data.csv.zip"
    )

    df_train = preprocess_data(df_train, "train")
    df_test = preprocess_data(df_test, "test")

    X_train, y_train = split_features_target(df_train, "default")
    X_test, y_test = split_features_target(df_test, "default")

    pipeline = pipeline_definition(X_train)
    grid_search = hyperparameter_optimization(pipeline, X_train, y_train)

    save_model(grid_search, "files/models/model.pkl.gz")

    train_metrics, train_cm = calculate_metrics(grid_search, X_train, y_train, "train")
    test_metrics, test_cm = calculate_metrics(grid_search, X_test, y_test, "test")

    metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    main()