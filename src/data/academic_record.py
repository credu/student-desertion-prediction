import pandas as pd


def getAcademicRecord():
    df = pd.read_csv("./src/data/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv")
    df['PROMEDIO'] = df['PROMEDIO'].str.replace(',', '.').astype(float)
    return df
