import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Carga los datos desde un CSV
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica de datos
    """
    df = df.sort_values(["store_id", "item_id", "date"])
    df = df.dropna()
    return df