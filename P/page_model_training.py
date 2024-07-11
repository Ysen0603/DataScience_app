import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from utils.model_utils import train_model, visualize_results, get_feature_importance, perform_cross_validation, save_model

def train_model_page():
    st.header("Entraînement du modèle")

    if st.session_state.data is not None:
        df = st.session_state.data

        st.subheader("Sélection des caractéristiques et de la cible")
        target_column = st.selectbox("Choisissez la colonne cible", df.columns)
        feature_columns = st.multiselect("Choisissez les colonnes de caractéristiques", [col for col in df.columns if col != target_column])

        if feature_columns:
            X = df[feature_columns]
            y = df[target_column]

            st.subheader("Sélection du type de modèle")
            model_type = st.radio("Choisissez le type de modèle", ["Classification", "Régression"])

            st.subheader("Sélection de l'algorithme")
            if model_type == "Classification":
                algorithm = st.selectbox("Choisissez l'algorithme de classification", ["Régression logistique", "Arbre de décision", "Forêt aléatoire"])
            else:
                algorithm = st.selectbox("Choisissez l'algorithme de régression", ["Régression linéaire", "Arbre de décision", "Forêt aléatoire"])

            test_size = st.slider("Taille de l'ensemble de test", 0.1, 0.5, 0.2, 0.05)
            use_grid_search = st.checkbox("Utiliser GridSearch pour optimiser les hyperparamètres")
            perform_cv = st.checkbox("Effectuer une validation croisée")

            if st.button("Entraîner le modèle"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model, y_pred = train_model(X_train, X_test, y_train, model_type, algorithm, use_grid_search)

                st.success("Modèle entraîné avec succès!")
                
                if model_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Précision du modèle : {accuracy:.2f}")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Erreur quadratique moyenne : {mse:.2f}")
                    st.write(f"Score R² : {r2:.2f}")

                # Visualisation des résultats
                st.subheader("Visualisation des résultats")
                fig = visualize_results(model, X_test, y_test, model_type)
                st.pyplot(fig)

                # Feature importance
                st.subheader("Importance des caractéristiques")
                fig_importance = get_feature_importance(model, feature_columns)
                if fig_importance:
                    st.pyplot(fig_importance)
                else:
                    st.write("L'importance des caractéristiques n'est pas disponible pour ce modèle.")

                # Validation croisée
                if perform_cv:
                    st.subheader("Résultats de la validation croisée")
                    cv_scores = perform_cross_validation(model, X, y)
                    st.write(f"Scores de validation croisée : {cv_scores}")
                    st.write(f"Score moyen : {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

                # Sauvegarde du modèle
                st.subheader("Sauvegarde du modèle")
                model_name = st.text_input("Nom du fichier pour sauvegarder le modèle", "mon_modele.joblib")
                if st.button("Sauvegarder le modèle"):
                    save_model(model, f"/models/saved_models/{model_name}", feature_columns, target_column, 
                               best_params=model.get_params() if use_grid_search else None)
                    st.success(f"Modèle sauvegardé sous le nom : {model_name}")

                st.session_state.model = model
                st.session_state.feature_columns = feature_columns
                st.session_state.target_column = target_column
        else:
            st.info("Veuillez sélectionner au moins une colonne de caractéristiques.")
    else:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")