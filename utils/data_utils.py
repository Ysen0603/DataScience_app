import pandas as pd
import chardet

def detect_encoding(file):
    raw_data = file.read(10000)  # Lire les premiers 10000 octets
    result = chardet.detect(raw_data)
    return result['encoding']

def load_excel_or_csv(file, **kwargs):
    try:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, **kwargs)
        elif file.name.endswith('.csv'):
            encoding = detect_encoding(file)
            file.seek(0)  # Remettre le curseur au début du fichier
            return pd.read_csv(file, encoding=encoding, **kwargs)
        else:
            raise ValueError("Le format de fichier n'est pas pris en charge. Utilisez .xlsx ou .csv")
    except Exception as e:
        raise IOError(f"Erreur lors du chargement du fichier : {str(e)}")