import streamlit as st
import pandas as pd
from models import desertion

st.title("Predicción de riesgo de deserción")
average_score = st.number_input(
    label="Promedio de calificaciones (0-10)",
    min_value=0.0,
    max_value=10.0
)
average_assistance = st.slider(
    label="Asistencia promedio (0-100)",
    min_value=0,
    max_value=100
)
max_subject_times = st.number_input(
    label="Veces maximas que ha repetido una sola materia",
    min_value=0,
    max_value=3
)
semester = st.number_input(
    label="Numero de Semestre",
    min_value=1,
)
subjects_failed = st.number_input(
    label="Total de materias reprobadas",
    min_value=0,
)
total_subjects = st.number_input(
    label="Numero de materias",
    min_value=1,
)

subjects_failed_ratio = subjects_failed / total_subjects

model = desertion.get_model()
data = pd.DataFrame([[
    average_score,
    average_assistance,
    max_subject_times,
    semester,
    subjects_failed_ratio,
    total_subjects
]], columns=desertion.columns)

prediccion = model.predict(data)[0]
probabilidades = model.predict_proba(data)[0]

st.subheader("Resultados")
st.text(f"Probabilidad de Permanencia: {probabilidades[0]:.2%}")
st.text(f"Probabilidad de Deserción: {probabilidades[1]:.2%}")
st.text(f"¿Deserción?: {"SÍ" if prediccion == 1 else "NO"}")
