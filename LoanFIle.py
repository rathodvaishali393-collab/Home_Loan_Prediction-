# LoanFIle.py
# This file loads and checks the Home Loan dataset used by the ML model.

import pandas as pd

DATASET_FILE = "loan_dataset.csv"

df = pd.read_csv(DATASET_FILE)

print("Home Loan Dataset loaded successfully.")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())
