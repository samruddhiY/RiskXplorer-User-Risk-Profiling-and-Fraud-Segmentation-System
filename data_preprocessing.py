# scripts/data_preprocessing.py

import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")
    print("Here's a quick sample:")
    print(df.head())
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Starting data cleaning...")

    # Drop columns not useful for modeling
    df = df.drop(['nameOrig', 'nameDest'], axis=1)

    # Encode 'type' column using one-hot encoding
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ Missing values found: {missing}. Filling with 0.")
        df = df.fillna(0)
    else:
        print("✅ No missing values found.")

    print(f"✅ Data cleaned. Final shape: {df.shape}")
    return df

def save_clean_data(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"📁 Cleaned data saved to: {out_path}")

if __name__ == "__main__":
    df_raw = load_data(r"F:\User_Risk_Profiling_Segmentation\PS_20174392719_1491204439457_log.csv")
    df_cleaned = clean_data(df_raw)
    save_clean_data(df_cleaned, "data/cleaned_transactions.csv")
