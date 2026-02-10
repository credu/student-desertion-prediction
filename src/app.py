import streamlit as st

main_page = st.Page("pages/home.py", title="Inicio", icon="⭐")
exploratory_analysis = st.Page(
    "pages/exploratory_analysis.py",
    title="Análisis exploratorio",
    icon="❄️",
    url_path="analisis-exploratorio"
)

model_metrics = st.Page(
    "pages/model_metrics.py",
    title="Metricas del modelo",
    icon="📐",
    url_path="metricas"
)

prediction_risk = st.Page(
    "pages/prediction_risk.py",
    title="Predicción de riesgo de deserción",
    icon="🧠"
)

pg = st.navigation([
    main_page,
    exploratory_analysis,
    model_metrics,
    prediction_risk
])

pg.run()
