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
        # Ajout des contrôles de taille dans la barre latérale
        st.sidebar.header("Paramètres du graphique")
        fig_width = st.sidebar.slider("Largeur du graphique", 5, 20, 10)
        fig_height = st.sidebar.slider("Hauteur du graphique", 3, 10, 6)


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
                                  ["Histogramme", "Diagramme en barres", "Boîte à moustaches", "Nuage de points", "Diagramme circulaire"])
        
        fig = None

        if graph_type == "Histogramme":
            selected_columns = st.multiselect("Choisissez une ou plusieurs colonnes", df.select_dtypes(include=[np.number]).columns)
            if selected_columns:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                for col in selected_columns:
                    sns.histplot(df[col], kde=True, ax=ax, label=col)
                plt.legend()
                plt.title("Histogramme")
            else:
                st.warning("Veuillez sélectionner au moins une colonne numérique.")

        elif graph_type == "Diagramme en barres":
            selected_column = st.selectbox("Choisissez une colonne", df.columns)
            if selected_column:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                df[selected_column].value_counts().plot(kind='bar', ax=ax)
                plt.title("Diagramme en barres")
            else:
                st.warning("Veuillez sélectionner une colonne.")

        elif graph_type == "Boîte à moustaches":
            selected_columns = st.multiselect("Choisissez une ou plusieurs colonnes", df.select_dtypes(include=[np.number]).columns)
            if selected_columns:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                df[selected_columns].boxplot(ax=ax)
                plt.title("Boîte à moustaches")
            else:
                st.warning("Veuillez sélectionner au moins une colonne numérique.")

        elif graph_type == "Nuage de points":
            x_col = st.selectbox("Choisissez la colonne pour l'axe X", df.select_dtypes(include=[np.number]).columns)
            y_cols = st.multiselect("Choisissez une ou plusieurs colonnes pour l'axe Y", df.select_dtypes(include=[np.number]).columns)
            if x_col and y_cols:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                for y_col in y_cols:
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, label=y_col)
                plt.legend()
                plt.title("Nuage de points")
            else:
                st.warning("Veuillez sélectionner une colonne pour l'axe X et au moins une colonne pour l'axe Y.")

        elif graph_type == "Diagramme circulaire":
            selected_column = st.selectbox("Choisissez une colonne", df.columns)
            if selected_column:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                data = df[selected_column].value_counts()
                wedges, _, autotexts = ax.pie(data.values, 
                                            labels=data.index, 
                                            autopct='%1.1f%%',
                                            textprops=dict(color="w"),
                                            startangle=90)
                ax.axis('equal')
                plt.setp(autotexts, size=8, weight="bold")
                plt.title(f"Distribution de {selected_column}")
                plt.legend(wedges, data.index,
                        title="Catégories",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1))
            else:
                st.warning("Veuillez sélectionner une colonne.")

        # Afficher le graphique
        if fig:
            st.pyplot(fig)
            
            # Option pour exporter le graphique
            if st.button("Exporter le graphique"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                
                st.download_button(
                    label="Télécharger le graphique",
                    data=buf,
                    file_name=f"{graph_type}.png",
                    mime="image/png"
                )

    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")