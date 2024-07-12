import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


def explore_data():
    st.header("Exploration des données")

    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Checkbox pour afficher les statistiques descriptives
        if st.checkbox("Afficher les statistiques descriptives"):
            st.subheader("Statistiques descriptives")
            st.write(df.describe())

        # Checkbox pour afficher les types de données
        if st.checkbox("Afficher les types de données"):
            st.subheader("Types de données")
            st.write(df.dtypes)

        # Checkbox pour afficher les valeurs manquantes
        if st.checkbox("Afficher les valeurs manquantes"):
            st.subheader("Valeurs manquantes")
            missing_data = df.isnull().sum()
            st.write(missing_data)

        # Corrélation entre les variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            if st.checkbox("Afficher la matrice de corrélation"):
                st.subheader("Corrélation entre les variables numériques")
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.write("Pas assez de colonnes numériques pour calculer la corrélation.")

        # Visualisation des données
        st.subheader("Visualisation des données")
        graph_type = st.selectbox("Choisissez le type de graphique", 
                                  ["Histogramme", "Diagramme en barres", "Boîte à moustaches", "Nuage de points"])
        
        fig = None  # Initialiser la variable fig
        
        if graph_type in ["Histogramme", "Diagramme en barres", "Boîte à moustaches"]:
            selected_column = st.selectbox("Choisissez une colonne", df.columns)
            
            if graph_type == "Histogramme":
                if df[selected_column].dtype in ['int64', 'float64']:
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_column], kde=True, ax=ax)
                else:
                    st.warning("Ce type de graphique nécessite une colonne numérique.")
            
            elif graph_type == "Diagramme en barres":
                fig, ax = plt.subplots()
                df[selected_column].value_counts().plot(kind='bar', ax=ax)
            
            elif graph_type == "Boîte à moustaches":
                if df[selected_column].dtype in ['int64', 'float64']:
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[selected_column], ax=ax)
                else:
                    st.warning("Ce type de graphique nécessite une colonne numérique.")
        
        elif graph_type == "Nuage de points":
            col1 = st.selectbox("Choisissez la première colonne", df.select_dtypes(include=[np.number]).columns)
            col2 = st.selectbox("Choisissez la deuxième colonne", df.select_dtypes(include=[np.number]).columns)
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col1, y=col2, ax=ax)

        # Afficher le graphique
        if fig:
            st.pyplot(fig)
            
            # Option pour exporter le graphique
            if st.button("Exporter le graphique"):
                # Sauvegarder le graphique dans un buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                
                # Offrir le téléchargement du fichier
                st.download_button(
                    label="Télécharger le graphique",
                    data=buf,
                    file_name=f"{graph_type}.png",
                    mime="image/png"
                )

    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")