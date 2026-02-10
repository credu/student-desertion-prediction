import streamlit as st
from models.desertion import get_model_metrics

accuracy, precision, recall, f1, conf_matrix = get_model_metrics()

st.title("Métricas del Modelo:")
st.write(f"**Accuracy:**  {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:**    {recall:.4f}")
st.write(f"**F1-Score:**  {f1:.4f}")

st.subheader("**Matriz de Confusión:**")
st.table(conf_matrix)
