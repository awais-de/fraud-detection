import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path="data/raw/creditcard.csv"):
    """
    Load the Credit Card Fraud Detection dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df



def preprocess_data(df, apply_smote=True, test_size=0.2, random_state=42):
    """
    Preprocess the fraud detection dataset.

    Steps:
    1. Convert Time -> Hour of day
    2. Scale Amount
    3. Train-test split
    4. Optionally apply SMOTE
    """
    # ---- Feature Engineering ----
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600) % 24

    # Scale Amount
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # ---- Split Features / Target ----
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ---- Handle Imbalance ----
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test



def summarize_data(df):
    """
    Print summary statistics of the dataset:
    - Shape
    - Missing values
    - Class distribution
    - Basic statistics for Amount and Time
    """
    print("📊 Dataset Summary")
    print("=" * 40)
    print(f"Shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False).head())
    print("\nClass distribution (fraud ratio):")
    print(df['Class'].value_counts(normalize=True) * 100)
    print("\nAmount statistics:")
    print(df['Amount'].describe())
    if 'Time' in df.columns:
        print("\nTime statistics:")
        print(df['Time'].describe())
    print("=" * 40)