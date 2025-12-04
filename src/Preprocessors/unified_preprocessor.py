from pandas import DataFrame
from pandas import concat

from src.Preprocessors.kaggle_preprocessor import KagglePreprocessor as KagglePreprocessor
from src.Preprocessors.synthetic_preprocessor import SyntheticPreprocessor as SyntheticProcessor


class UnifiedPreprocessor:
    def __init__(self, synthetic_preprocessor=None, kaggle_preprocessor=None):
        self.synthetic_preprocessor = synthetic_preprocessor or SyntheticProcessor()
        self.kaggle_preprocessor = kaggle_preprocessor or KagglePreprocessor()
        self.expected_columns = None  # Columns after training preprocessing

    def preprocess(self,
                   synthetic_source: str,
                   kaggle_source: str,
                   shuffle: bool = True) -> DataFrame:

        print("🚀 Running Synthetic Preprocessor...")
        syn = self.synthetic_preprocessor.preprocess(synthetic_source)

        print("🚀 Running Kaggle Preprocessor...")
        kaggle = self.kaggle_preprocessor.preprocess(kaggle_source)

        # ---------------------------------------------------------
        # ALIGN COLUMNS
        # ---------------------------------------------------------
        print("🔧 Aligning columns...")

        syn_cols = set(syn.columns)
        kaggle_cols = set(kaggle.columns)

        # Columns that exist ONLY in one dataset
        only_syn = syn_cols - kaggle_cols
        only_kaggle = kaggle_cols - syn_cols

        # Add missing columns as zeros (safe for one-hot + numeric)
        for col in only_syn:
            kaggle[col] = 0

        for col in only_kaggle:
            syn[col] = 0

        # Ensure same column order
        syn = syn.sort_index(axis=1)
        kaggle = kaggle.sort_index(axis=1)

        # ---------------------------------------------------------
        # CONCATENATE
        # ---------------------------------------------------------
        print("📦 Merging datasets...")
        df = concat([syn, kaggle], ignore_index=True)

        # Create unified label column
        if "Class" in df.columns:
            df["label"] = df["Class"]
        elif "anomaly" in df.columns:
            df["label"] = df["anomaly"]
        else:
            raise KeyError("Neither 'Class' nor 'anomaly' exists in df.")

        # Remove dataset-specific label columns
        for col in ["Class", "anomaly"]:
            if col in df.columns:
                df = df.drop(columns=col)

        # print("After label normalization:", df.columns.tolist())

        # ---------------------------------------------------------
        # OPTIONAL: SHUFFLE
        # ---------------------------------------------------------
        if shuffle:
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Save expected columns for runtime API
        self.expected_columns = [c for c in df.columns if c != "label"]

        print("✅ Unified preprocessing complete.")
        print(f"📊 Final shape: {df.shape}\n")
        return df

    # -------------------------
    # Runtime preprocessing (for API)
    # -------------------------
    def preprocess_runtime(self, df: DataFrame) -> DataFrame:
        """
        Preprocess a DataFrame from API input to match training columns.
        """
        if self.expected_columns is None:
            raise RuntimeError("UnifiedPreprocessor expected_columns not set. Call preprocess() first.")

        df = df.copy()

        # Synthetic preprocessing steps
        df = self.synthetic_preprocessor.clean_amounts(df)
        df = self.synthetic_preprocessor.encode_categoricals(df)
        df = self.synthetic_preprocessor.scale_numeric(df)
        df = self.synthetic_preprocessor.feature_engineering(df)

        # Kaggle preprocessing steps
        df = self.kaggle_preprocessor.clean_amounts(df)
        df = self.kaggle_preprocessor.encode_categoricals(df)
        df = self.kaggle_preprocessor.scale_numeric(df)
        df = self.kaggle_preprocessor.feature_engineering(df)

        # Add missing columns
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep only expected columns (exclude label)
        df = df[[col for col in self.expected_columns if col in df.columns]]

        return df

    # -------------------------
    # BasePreprocessor abstract methods
    # -------------------------
    def clean_amounts(self, df: DataFrame) -> DataFrame:
        """Pass-through; handled by child preprocessors."""
        return df

    def encode_categoricals(self, df: DataFrame, categorical_cols: list[str] = None) -> DataFrame:
        """Pass-through; handled by child preprocessors."""
        return df

    def scale_numeric(self, df: DataFrame) -> DataFrame:
        """Pass-through; handled by child preprocessors."""
        return df

    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """Pass-through; handled by child preprocessors."""
        return df
