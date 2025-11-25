import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.BaseClasses.base_preprocessor import BasePreprocessor


class SyntheticPreprocessor(BasePreprocessor):
    """
    Preprocess synthetic anomaly dataset.
    Steps:
      1. Load CSV
      2. Convert timestamp
      3. Clean numeric anomalies
      4. One-hot encode categorical features
      5. Scale numeric features
    """

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.raw_dataframe = None
        self.processed_dataframe = None

    def preprocess(self, source: str) -> pd.DataFrame:
        print(f"📦 Loading Synthetic dataset: {source}")
        df = self.load(source)
        self.raw_dataframe = df.copy()

        print("⏰ Converting timestamp column...")
        df = self.convert_timestamp(df)

        print("🧹 Cleaning numeric anomalies...")
        df = self.clean_amounts(df)

        print("🎨 Encoding categorical columns...")
        df = self.encode_categoricals(df)

        print("📊 Scaling numeric features...")
        df = self.scale_numeric(df)
        self.processed_dataframe = df

        print("⚙️ Performing feature engineering...")
        df = self.feature_engineering(df)

        print("✅ Synthetic preprocessing complete.")
        print(f"📈 Dataset shape: {df.shape}\n")
        return df

    # -------------------------
    # Load CSV
    # -------------------------

    # -------------------------
    # Convert timestamp
    # -------------------------
    def convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    # -------------------------
    # Clean numeric anomalies
    # -------------------------
    def clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        df["amount"] = df["amount"].apply(lambda x: max(x, 0))
        return df

    # -------------------------
    # One-hot encode categorical
    # -------------------------
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "category" in df.columns:
            df = pd.get_dummies(df, columns=["category"], prefix="cat")
        return df

    # -------------------------
    # Scale numeric features
    # -------------------------
    def scale_numeric(self, df):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Remove columns that should never be scaled
        exclude = ['timestamp', 'status', 'label']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        # print(f"Scaling numeric columns: {numeric_cols}")

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df
