import pandas as pd

class UnifiedPreprocessor:
    def __init__(self, synthetic_processor, kaggle_processor):
        self.synthetic_processor = synthetic_processor
        self.kaggle_processor = kaggle_processor

    def preprocess(self,
                   synthetic_source: str,
                   kaggle_source: str,
                   shuffle: bool = True) -> pd.DataFrame:

        print("🚀 Running Synthetic Preprocessor...")
        syn = self.synthetic_processor.preprocess(synthetic_source)

        print("🚀 Running Kaggle Preprocessor...")
        kaggle = self.kaggle_processor.preprocess(kaggle_source)

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
        df = pd.concat([syn, kaggle], ignore_index=True)

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

        print("✅ Unified preprocessing complete.")
        print(f"📊 Final shape: {df.shape}\n")
        return df
