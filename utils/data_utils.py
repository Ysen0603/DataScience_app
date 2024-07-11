import pandas as pd

def load_excel_or_csv(file):
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        raise ValueError("Le format de fichier n'est pas pris en charge. Utilisez .xlsx ou .csv")