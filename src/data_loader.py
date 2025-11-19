import pandas as pd
from pathlib import Path


def load_csv(file_path: str) -> pd.DataFrame:
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    # Load CSV into DataFrame
    dataframe = pd.read_csv(file_path)
    # Return DataFrame
    return dataframe
