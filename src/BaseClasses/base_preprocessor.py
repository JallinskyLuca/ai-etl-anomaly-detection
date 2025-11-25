import abc
from pathlib import Path

from pandas import DataFrame

from src.data_loader import load_csv


class BasePreprocessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def clean_amounts(self, df: DataFrame) -> DataFrame:
        """Subclasses must implement this."""
        raise NotImplementedError('Subclasses must implement data_cleaner method.')

    @abc.abstractmethod
    def encode_categoricals(self, df: DataFrame, categorical_cols: list[str] = None) -> DataFrame:
        """Subclasses must implement this."""
        raise NotImplementedError('Subclasses must implement encode_categoricals method.')

    def feature_engineering(self, df: DataFrame) -> DataFrame:
        # Handle timestamps for both datasets
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        else:
            df['hour_of_day'] = 0
            df['day_of_week'] = 0

        # Synthetic dataset has actual customer IDs
        # Kaggle we generate customer_id in clean_data()
        if 'customer_id' in df.columns and 'timestamp' in df.columns:
            df['time_since_last'] = df.groupby('customer_id')['timestamp'].diff().dt.total_seconds()

            # Fill the first transaction per customer with 0
            df['time_since_last'] = df['time_since_last'].fillna(0)
        else:
            df['time_since_last'] = 0

        return df

    def load(self, source: str) -> DataFrame:
        import os
        from pathlib import Path

        if not os.path.exists(source):
            project_root = Path(__file__).resolve().parents[2]
            alt_path = project_root / 'data' / 'raw' / source
            if alt_path.exists():
                source = alt_path

        return load_csv(source)

    @abc.abstractmethod
    def preprocess(self, filepath: Path) -> DataFrame:
        """Subclasses must implement this."""
        raise NotImplementedError('Subclasses must implement preprocess method.')

    @abc.abstractmethod
    def scale_numeric(self, df: DataFrame) -> DataFrame:
        """Subclasses must implement this."""
        raise NotImplementedError('Subclasses must implement scale_numeric method.')
