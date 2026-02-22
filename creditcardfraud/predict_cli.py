# predict_cli.py
import os
import json
import joblib
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(PROJECT_DIR, "artifacts")

MODEL_PATH = os.path.join(ART_DIR, "model.joblib")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
COLS_PATH = os.path.join(ART_DIR, "feature_columns.json")
PROFILES_PATH = os.path.join(ART_DIR, "customer_profiles.csv")


def read_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Enter a valid number.")


def read_hour():
    while True:
        try:
            h = int(input("Enter Transaction Hour (0–23): ").strip())
            if 0 <= h <= 23:
                return h
            print("Hour must be between 0 and 23.")
        except ValueError:
            print("Enter a valid integer.")


def read_binary(prompt):
    while True:
        val = input(prompt + " (y/n): ").strip().lower()
        if val in ["y", "n"]:
            return "Yes" if val == "y" else "No"
        print("Enter y or n.")


def main():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    profiles = pd.read_csv(PROFILES_PATH)

    print("Fraud Prediction System Ready")

    while True:
        cust_id = input("\nEnter Customer_ID: ").strip()

        match = profiles[profiles["Customer_ID"] == cust_id]
        if match.empty:
            print("Customer not found.")
            continue

        customer = match.iloc[0]

        amount = read_float("Enter Transaction Amount: ")
        hour = read_hour()
        is_new = read_binary("Is this a new device")

        avg = customer["Customer_Avg_Amount"]
        std = customer["Customer_Amount_Std"]
        peak = customer["Normal_Peak_Hour"]

        amount_dev = amount - avg
        amount_ratio = amount / avg if avg != 0 else 0
        hour_diff = abs(hour - peak)

        # Minimal feature input
        input_data = {
            "Transaction_Amount": amount,
            "Customer_Avg_Amount": avg,
            "Customer_Amount_Std": std,
            "Amount_Deviation": amount_dev,
            "Amount_Ratio": amount_ratio,
            "Transaction_Hour": hour,
            "Hour_Difference": hour_diff,
            "Country": customer["Country"],
            "Merchant_Category": "Groceries",  # fixed default
            "Location_Type": "Domestic",       # fixed default
            "Device_Type": "Mobile",           # fixed default
            "Is_New_Device": is_new
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            print(f"\nFRAUD (probability={probability:.4f})")
        else:
            print(f"\nNOT FRAUD (probability={probability:.4f})")

        again = input("Check another? (y/n): ").lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()