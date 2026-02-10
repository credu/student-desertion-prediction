import streamlit as st
from data.academic_record import getAcademicRecord
import matplotlib.pyplot as plt
import seaborn as sns
from models import desertion

df = getAcademicRecord()
st.title("Analisis exploratorio")
st.text("Datos")
st.dataframe(df)

df_importancia = desertion.get_feature_importances()

fig, ax = plt.subplots()

sns.barplot(
    x="Importancia",
    y="Feature",
    data=df_importancia,
    hue="Feature",
    palette="viridis"
)

st.pyplot(fig)
