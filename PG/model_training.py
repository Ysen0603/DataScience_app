import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from utils.model_utils import train_model, visualize_results, get_feature_importance, perform_cross_validation
import joblib
import tempfile

def train_model_page():
    st.header("Model Training")

    if st.session_state.data is not None:
        df = st.session_state.data

        st.subheader("Feature and Target Selection")
        target_column = st.selectbox("Choose the target column", df.columns)
        feature_columns = st.multiselect("Choose feature columns", [col for col in df.columns if col != target_column])

        if feature_columns:
            X = df[feature_columns]
            y = df[target_column]

            st.subheader("Model Type Selection")
            model_type = st.radio("Choose model type", ["Classification", "Regression"])

            st.subheader("Algorithm Selection")
            if model_type == "Classification":
                algorithm = st.selectbox("Choose classification algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"])
            else:
                algorithm = st.selectbox("Choose regression algorithm", ["Linear Regression", "Decision Tree", "Random Forest"])

            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            use_grid_search = st.checkbox("Use GridSearch for hyperparameter optimization")
            perform_cv = st.checkbox("Perform cross-validation")

            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model, y_pred = train_model(X_train, X_test, y_train, model_type, algorithm, use_grid_search)

                st.success("Model trained successfully!")
                
                if model_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Model Accuracy: {accuracy:.2f}")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"R² Score: {r2:.2f}")

                # Visualize results
                st.subheader("Visualize Results")
                fig = visualize_results(model, X_test, y_test, model_type)
                st.pyplot(fig)

                # Feature importance
                st.subheader("Feature Importance")
                fig_importance = get_feature_importance(model, feature_columns)
                if fig_importance:
                    st.pyplot(fig_importance)
                else:
                    st.write("Feature importance is not available for this model.")

                # Cross-validation
                if perform_cv:
                    st.subheader("Cross-Validation Results")
                    cv_scores = perform_cross_validation(model, X, y)
                    st.write(f"Cross-validation scores: {cv_scores}")
                    st.write(f"Mean score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

                # Save model
                st.header("Save Model on the Next Page")
                model_name = st.text_input("File name to save the model", "my_model.joblib")
                if st.button("Download Model"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                        joblib.dump(model, tmp_file.name)
                        tmp_file_path = tmp_file.name

                    with open(tmp_file_path, "rb") as f:
                        st.download_button(label="Download Model", data=f, file_name=model_name)

                st.session_state.model = model
                st.session_state.feature_columns = feature_columns
                st.session_state.target_column = target_column
        else:
            st.info("Please select at least one feature column.")
    else:
        st.warning("No data has been loaded. Please load a file first in the 'Data Loading' section.")
