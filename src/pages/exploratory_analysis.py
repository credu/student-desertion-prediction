import streamlit as st
from data.academic_record import getAcademicRecord

df = getAcademicRecord()
st.title("Analisis exploratorio")
st.text("Datos")
st.dataframe(df)
