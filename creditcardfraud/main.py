# main.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "behavioral_fraud_dataset_realistic_time.csv")

ART_DIR = os.path.join(PROJECT_DIR, "artifacts")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Transaction_Timestamp"] = pd.to_datetime(df["Transaction_Timestamp"], errors="coerce")
    if df["Transaction_Timestamp"].isna().any():
        raise ValueError("Some Transaction_Timestamp values could not be parsed. Check your CSV format.")

    df["Transaction_Hour"] = df["Transaction_Timestamp"].dt.hour.astype(int)

    df["Amount_Deviation"] = df["Transaction_Amount"] - df["Customer_Avg_Amount"]
    df["Amount_Ratio"] = df["Transaction_Amount"] / df["Customer_Avg_Amount"].replace(0, np.nan)
    df["Amount_Ratio"] = df["Amount_Ratio"].fillna(0.0)

    df["Hour_Difference"] = (df["Transaction_Hour"] - df["Normal_Peak_Hour"]).abs()

    features = pd.DataFrame({
        "Transaction_Amount": df["Transaction_Amount"].astype(float),
        "Customer_Avg_Amount": df["Customer_Avg_Amount"].astype(float),
        "Customer_Amount_Std": df["Customer_Amount_Std"].astype(float),
        "Amount_Deviation": df["Amount_Deviation"].astype(float),
        "Amount_Ratio": df["Amount_Ratio"].astype(float),
        "Transaction_Hour": df["Transaction_Hour"].astype(int),
        "Hour_Difference": df["Hour_Difference"].astype(int),
        "Country": df["Country"].astype(str),
        "Merchant_Category": df["Merchant_Category"].astype(str),
        "Location_Type": df["Location_Type"].astype(str),
        "Device_Type": df["Device_Type"].astype(str),
        "Is_New_Device": df["Is_New_Device"].astype(str),
    })

    # One-hot encode categoricals
    features = pd.get_dummies(features, drop_first=True)
    return features


def save_customer_profiles(df: pd.DataFrame, out_csv_path: str):
    # One row per customer for inference lookups
    prof = (
        df.groupby("Customer_ID", as_index=False)
          .agg({
              "Country": "first",
              "Customer_Avg_Amount": "first",
              "Customer_Amount_Std": "first",
              "Normal_Peak_Hour": "first"
          })
    )
    prof.to_csv(out_csv_path, index=False)


def main():
    df = pd.read_csv(DATA_PATH)

    if "Fraud_Label" not in df.columns:
        raise ValueError("Fraud_Label column not found in dataset.")
    if "Customer_ID" not in df.columns:
        raise ValueError("Customer_ID column not found in dataset.")

    y = df["Fraud_Label"].astype(int).to_numpy()

    X = build_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric + encoded features (safe + standard)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.to_numpy())
    X_test_s = scaler.transform(X_test.to_numpy())

    # Model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4)

    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion Matrix [ [TN FP] [FN TP] ]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    # Save artifacts
    joblib.dump(model, os.path.join(ART_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(ART_DIR, "scaler.joblib"))

    with open(os.path.join(ART_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f)

    save_customer_profiles(df, os.path.join(ART_DIR, "customer_profiles.csv"))

    print("Training complete.")
    print(f"Saved model + scaler + columns to: {ART_DIR}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()