import pandas as pd
import streamlit as st


@st.cache_data
def getAcademicRecord():
    df = pd.read_csv("./src/data/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv")
    df["PROMEDIO"] = df["PROMEDIO"].str.replace(",", ".").astype(float)
    return df


@st.cache_data
def get_students_data():
    df = getAcademicRecord()
    mask = df["PERIODO"].str.startswith("2025 -")
    active_students_id = df[mask]["ESTUDIANTE"].unique()
    df_features = df[~df["PERIODO"].str.contains("ING")]
    student_data = df_features.groupby("ESTUDIANTE").agg({
        "PROMEDIO": "mean",
        "ASISTENCIA": "mean",
        "NO. VEZ": "max",
        "NIVEL": "max",
        "ESTADO": [lambda x: (x == "REPROBADA").sum(), "count"]
    })

    student_data.columns = [
        "PROMEDIO",
        "ASISTENCIA",
        "NO. VEZ",
        "NIVEL_MAXIMO",
        "MATERIAS_REPROBADAS",
        "TOTAL_MATERIAS"
    ]

    student_data["RATIO_REPROBADAS"] = \
        student_data["MATERIAS_REPROBADAS"] / student_data["TOTAL_MATERIAS"]

    student_data["DESERTO"] = student_data.index.map(
        lambda x: 0 if x in active_students_id else 1
    )
    return student_data
