import pandas as pd

DATA_PATH = "test_invoices.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n===== DATA TYPES =====")0
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATES =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== FIRST 5 ROWS =====")
print(df.head())