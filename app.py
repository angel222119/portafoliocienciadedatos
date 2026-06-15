import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from textblob import TextBlob

# --------------------------------
# CONFIGURACIÓN
# --------------------------------

st.set_page_config(
    page_title="Portafolio de Ciencia de Datos",
    layout="wide"
)

# --------------------------------
# DATASET
# --------------------------------

@st.cache_data
def cargar_datos():
    return sns.load_dataset("titanic")

df = cargar_datos()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("📊 Menú Principal")

opcion = st.sidebar.radio(
    "Seleccione una opción",
    [
        "Inicio",
        "Análisis Exploratorio",
        "Aprendizaje Automático",
        "Sistema de Recomendación",
        "Carga de Archivos",
        "Análisis de Sentimientos"
    ]
)

# ==================================================
# INICIO
# ==================================================

if opcion == "Inicio":

    st.title("📊 Portafolio Profesional de Ciencia de Datos")

    st.header("Luis Angel Escobar Constanza")
    st.subheader("Código: SMSS144922")

    st.markdown("""
    **Carrera:** Ingeniería en Sistemas y Redes Informáticas

    **Asignatura:** Técnica Electiva I - Ciencia de Datos
    """)

    st.write("""
    Mi nombre es Luis Angel Escobar Constanza, estudiante de Ingeniería en Sistemas y Redes Informáticas.
    Este portafolio presenta técnicas de análisis exploratorio, aprendizaje automático,
    sistemas de recomendación y análisis de sentimientos utilizando herramientas modernas
    de Ciencia de Datos.
    """)

    st.info("")

    st.video("")

# ==================================================
# ANALISIS EXPLORATORIO
# ==================================================

elif opcion == "Análisis Exploratorio":

    submenu = st.selectbox(
        "Seleccione una opción",
        [
            "Descripción Dataset",
            "Descripción Campos",
            "Navegador Dataset",
            "Graficador",
            "Hipótesis"
        ]
    )

    if submenu == "Descripción Dataset":

        st.header("Descripción del Dataset")

        st.write(df.head())

        st.write("Filas:", df.shape[0])
        st.write("Columnas:", df.shape[1])

        st.write(df.describe())

    elif submenu == "Descripción Campos":

        campo = st.selectbox("Seleccione un campo", df.columns)

        st.subheader(campo)

        if pd.api.types.is_numeric_dtype(df[campo]):
            st.write(df[campo].describe())
        else:
            st.write(df[campo].value_counts())

    elif submenu == "Navegador Dataset":

        st.dataframe(df)

    elif submenu == "Graficador":

        campo = st.selectbox("Seleccione columna", df.columns)

        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(df[campo]):
            sns.histplot(df[campo], kde=True, ax=ax)
        else:
            df[campo].value_counts().plot(kind="bar", ax=ax)

        st.pyplot(fig)

    elif submenu == "Hipótesis":

        hipotesis = st.selectbox(
            "Seleccione hipótesis",
            [
                "Las mujeres sobrevivieron más",
                "Primera clase sobrevivió más"
            ]
        )

        if hipotesis == "Las mujeres sobrevivieron más":

            st.subheader("Hipótesis 1")

            fig = px.histogram(
                df,
                x="sex",
                color="survived",
                barmode="group"
            )

            st.plotly_chart(fig)

            st.success("""
            Conclusión:
            Las mujeres presentaron una mayor tasa de supervivencia
            que los hombres.
            """)

        if hipotesis == "Primera clase sobrevivió más":

            st.subheader("Hipótesis 2")

            fig = px.bar(
                df.groupby("pclass")["survived"].mean().reset_index(),
                x="pclass",
                y="survived"
            )

            st.plotly_chart(fig)

            st.success("""
            Conclusión:
            Los pasajeros de primera clase sobrevivieron más.
            """)

# ==================================================
# MACHINE LEARNING
# ==================================================

elif opcion == "Aprendizaje Automático":

    st.header("Aprendizaje Automático")

    data_ml = df.copy()

    columnas = ["pclass", "age", "fare"]

    data_ml = data_ml[columnas + ["survived"]]

    data_ml = data_ml.dropna()

    algoritmo = st.selectbox(
        "Seleccione algoritmo",
        [
            "Regresión Logística",
            "Árbol de Decisión"
        ]
    )

    porcentaje = st.slider(
        "Porcentaje de entrenamiento",
        50,
        90,
        80
    )

    X = data_ml[columnas]
    y = data_ml["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=porcentaje/100,
        random_state=42
    )

    if algoritmo == "Regresión Logística":
        modelo = LogisticRegression(max_iter=1000)

    else:
        modelo = DecisionTreeClassifier()

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("Entrenamiento (%)", porcentaje)

    resultados = pd.DataFrame({
        "Real": y_test,
        "Predicción": pred
    })

    st.dataframe(resultados.head(20))

# ==================================================
# SISTEMA RECOMENDACION
# ==================================================

elif opcion == "Sistema de Recomendación":

    st.header("🎬 Sistema de Recomendación de Películas")

    peliculas = {
        "Titanic":["Avatar","The Notebook","Pearl Harbor"],
        "Avatar":["Titanic","Interstellar","Gravity"],
        "Joker":["Batman","The Dark Knight","Logan"],
        "Interstellar":["Gravity","Avatar","The Martian"]
    }

    pelicula = st.selectbox(
        "Seleccione película",
        list(peliculas.keys())
    )

    st.subheader("Recomendaciones")

    for p in peliculas[pelicula]:
        st.write("✅", p)

# ==================================================
# CARGA DE ARCHIVOS
# ==================================================

elif opcion == "Carga de Archivos":

    st.header("📂 Análisis de Datos por Archivo")

    archivo = st.file_uploader(
        "Subir CSV",
        type=["csv"]
    )

    if archivo is not None:

        datos = pd.read_csv(archivo)

        st.dataframe(datos.head())

        columna = st.selectbox(
            "Seleccione columna",
            datos.columns
        )

        if pd.api.types.is_numeric_dtype(datos[columna]):

            fig = px.histogram(
                datos,
                x=columna
            )

            st.plotly_chart(fig)

        else:

            fig = px.bar(
                datos[columna].value_counts().reset_index(),
                x="index",
                y=columna
            )

            st.plotly_chart(fig)

# ==================================================
# SENTIMIENTOS
# ==================================================

elif opcion == "Análisis de Sentimientos":

    st.header("😊 Análisis de Sentimientos")

    texto = st.text_area(
        "Ingrese una opinión"
    )

    if st.button("Analizar"):

        analisis = TextBlob(texto)

        polaridad = analisis.sentiment.polarity

        st.write("Polaridad:", polaridad)

        if polaridad > 0:
            st.success("Sentimiento Positivo")

        elif polaridad < 0:
            st.error("Sentimiento Negativo")

        else:
            st.warning("Sentimiento Neutral")