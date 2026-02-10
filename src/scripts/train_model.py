import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

MODEL_PATH = "src/models/desertion.pkl"


def get_data():
    df = pd.read_csv("./src/data/REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv")
    df['PROMEDIO'] = df['PROMEDIO'].str.replace(',', '.').astype(float)
    return df


def get_students_data(df: pd.DataFrame):
    mask = df["PERIODO"].str.startswith('2025 -')
    active_students_id = df[mask]["ESTUDIANTE"].unique()
    df_features = df[~df['PERIODO'].str.contains('ING')]
    student_data = df_features.groupby('ESTUDIANTE').agg({
        'PROMEDIO': 'mean',
        'ASISTENCIA': 'mean',
        'NO. VEZ': 'max',
        'NIVEL': 'max',
        'ESTADO': [lambda x: (x == 'REPROBADA').sum(), 'count']
    })

    student_data.columns = [
        "PROMEDIO",
        "ASISTENCIA",
        "NO. VEZ",
        "NIVEL_MAXIMO",
        "MATERIAS_REPROBADAS",
        "TOTAL_MATERIAS"
    ]

    student_data['RATIO_REPROBADAS'] = \
        student_data['MATERIAS_REPROBADAS'] / student_data['TOTAL_MATERIAS']

    student_data['DESERTO'] = student_data.index.map(
        lambda x: 0 if x in active_students_id else 1
    )
    return student_data


def train_model(df: pd.DataFrame):
    features = [
        "PROMEDIO",
        "ASISTENCIA",
        "NO. VEZ",
        "NIVEL_MAXIMO",
        "RATIO_REPROBADAS",
        "TOTAL_MATERIAS"
    ]

    X = df[features]
    y = df['DESERTO']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model):
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    except Exception:
        print("Error: Error not handled")


def main():
    df = get_data()
    students_data = get_students_data(df)
    model, X_test, y_test = train_model(students_data)
    save_model([model, X_test, y_test])
    print("Modelo guardado")


if __name__ == "__main__":
    main()
