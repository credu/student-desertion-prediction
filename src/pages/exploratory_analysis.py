import streamlit as st
from data.academic_record import getAcademicRecord, get_students_data
import matplotlib.pyplot as plt
import seaborn as sns
from models import desertion

student_data = get_students_data()
academic_record_df = getAcademicRecord()
feature_importances_df = desertion.get_feature_importances()

st.title("🔍 Analisis exploratorio de Datos")

c1, c2, c3 = st.columns(3)
c1.metric("Total Estudiantes", len(student_data))
c2.metric("Tasa de Deserción", f"{student_data["DESERTO"].mean():.1%}")
c3.metric("Promedio General", f"{student_data["PROMEDIO"].mean():.2f}")

with st.expander("Ver datos históricos"):
    st.dataframe(academic_record_df, use_container_width=True)

with st.expander("Variables más importantes", expanded=True):
    st.subheader("Variables más importantes para la predicción")
    st.write(
        "Este gráfico muestra qué tanto influye cada factor en la decisión de "
        "un estudiante de abandonar sus estudios."
    )

    fig, ax = plt.subplots()
    sns.barplot(
        x="Importancia",
        y="Feature",
        data=feature_importances_df,
        hue="Feature",
        palette="viridis",
    )
    st.pyplot(fig)

st.subheader("¿Cómo influyen las notas y la asistencia?")
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots()
    sns.boxplot(
        x="DESERTO",
        y="PROMEDIO",
        data=student_data,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Distribución de Promedios")
    st.pyplot(fig)

with col_b:
    fig, ax = plt.subplots()
    sns.boxplot(
        x="DESERTO",
        y="ASISTENCIA",
        data=student_data,
        palette="Set2",
        ax=ax
    )
    ax.set_title("Distribución de Asistencia")
    st.pyplot(fig)

st.subheader("Mapa de Calor de Variables")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(student_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
