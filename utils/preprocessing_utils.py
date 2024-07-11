import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def encode_categorical(df, columns, method):
    if method == "One-Hot Encoding":
        return pd.get_dummies(df, columns=columns)
    elif method == "Label Encoding":
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def normalize_data(df, columns, method):
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_missing_values(df, columns, method):
    if method == "Supprimer":
        return df.dropna(subset=columns)
    elif method == "Moyenne":
        imputer = SimpleImputer(strategy='mean')
    elif method == "Médiane":
        imputer = SimpleImputer(strategy='median')
    elif method == "Valeur la plus fréquente":
        imputer = SimpleImputer(strategy='most_frequent')
    
    df[columns] = imputer.fit_transform(df[columns])
    return df