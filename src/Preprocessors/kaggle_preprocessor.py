import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.BaseClasses.base_preprocessor import BasePreprocessor


class KagglePreprocessor(BasePreprocessor):
    """
    Preprocess Kaggle credit card fraud dataset.
    Steps:
      1. Load CSV
      2. Convert Time column to datetime
      3. Clean amounts
      4. Scale numeric features
      5. Return processed DataFrame
    """

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.raw_dataframe = None
        self.processed_dataframe = None

    def preprocess(self, source: str) -> pd.DataFrame:
        print(f"📦 Loading Kaggle dataset: {source}")
        df = self.load(source)
        self.raw_dataframe = df.copy()

        print("⏰ Converting Time column to timestamp...")
        df = self.convert_datetime(df)

        print("🧹 Cleaning numeric anomalies...")
        df = self.clean_amounts(df)

        print("📊 Scaling numeric features...")
        df = self.scale_numeric(df)
        self.processed_dataframe = df

        print("⚙️ Performing feature engineering...")
        df = self.feature_engineering(df)

        print("✅ Kaggle preprocessing complete.")
        print(f"📈 Dataset shape: {df.shape}\n")
        return df

    # -------------------------
    # Load CSV
    # -------------------------

    # -------------------------
    # Convert Time column
    # -------------------------
    def convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["Time"], unit="s", origin="2023-01-01")
            df.drop(columns=["Time"], inplace=True)

        # Create dummy customer_id
        df["customer_id"] = df.index.astype(int)
        return df

    # -------------------------
    # Clean numeric anomalies
    # -------------------------
    def clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Amount" in df.columns:
            df["Amount"] = df["Amount"].apply(lambda x: max(x, 0))
        return df

    # -------------------------
    # One-hot encode categorical
    # -------------------------
    def encode_categoricals(self, df: pd.DataFrame, categorical_cols: list[str] = None) -> pd.DataFrame:
        """No categorical features in Kaggle dataset."""
        return df

    # -------------------------
    # Scale numeric features
    # -------------------------
    def scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        # Kaggle feature names:
        # V1..V28 = PCA components
        # Amount = original transaction amount
        # Time = seconds from first transaction (we do NOT scale this — we replace it in feature engineering)

        # Select numeric features explicitly
        numeric_cols = [c for c in df.columns if c.startswith("V")]  # PCA components
        numeric_cols.append("Amount")

        # Exclude target label
        exclude = ["Class", "label", "status"]
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        print(f"[Kaggle] Scaling numeric columns: {numeric_cols}")

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df
