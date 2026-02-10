import pickle
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

columns = [
    "PROMEDIO",
    "ASISTENCIA",
    "NO. VEZ",
    "NIVEL_MAXIMO",
    "RATIO_REPROBADAS",
    "TOTAL_MATERIAS"
]


@st.cache_resource
def __unpack_model_data() -> tuple[RandomForestClassifier, any, any]:
    try:
        with open('src/models/desertion.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        print("Error: Error not handled")


def get_model() -> RandomForestClassifier:
    return __unpack_model_data()[0]


@st.cache_resource
def get_model_metrics():
    model, X_test, y_test = __unpack_model_data()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, conf_matrix
