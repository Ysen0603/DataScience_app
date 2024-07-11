import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import streamlit as st

def train_model(X_train, X_test, y_train, model_type, algorithm, use_grid_search=False):
    if model_type == "Classification":
        if algorithm == "Régression logistique":
            model = LogisticRegression()
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        elif algorithm == "Arbre de décision":
            model = DecisionTreeClassifier()
            param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
        elif algorithm == "Forêt aléatoire":
            model = RandomForestClassifier()
            param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10, None]}
    else:
        if algorithm == "Régression linéaire":
            model = LinearRegression()
            param_grid = {}  # Pas de paramètres à optimiser pour la régression linéaire
        elif algorithm == "Arbre de décision":
            model = DecisionTreeRegressor()
            param_grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
        elif algorithm == "Forêt aléatoire":
            model = RandomForestRegressor()
            param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10, None]}

    if use_grid_search and param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if model_type == "Classification" else 'neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.write("Meilleurs paramètres trouvés:", grid_search.best_params_)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    return model, y_pred

def visualize_results(model, X_test, y_test, model_type):
    if model_type == "Classification":
        # Courbe ROC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt
    else:
        # Graphique des résidus
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.axhline(y=0, color='r', linestyle='--')
        return plt

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return plt
    return None

def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores

def save_model(model, filename, feature_columns, target_column, best_params=None):
    model_info = {
        'model': model,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'best_params': best_params
    }
    joblib.dump(model_info, filename)

def load_model(filename):
    return joblib.load(filename)