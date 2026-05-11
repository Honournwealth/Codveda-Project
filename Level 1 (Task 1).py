import pandas as pd
import numpy as np

# =========================
# 0. DISPLAY SETTINGS
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("iris.csv")

# Standardize column names
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

print("\n================ RAW DATA INSPECTION ================")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nInitial Dataset Shape:", df.shape)

# =========================
# 2. CHECK DUPLICATES
# =========================
print("\n================ FULL ROW DUPLICATES ================")

duplicates = df.duplicated()

print("Total Duplicates Found:", duplicates.sum())

print("\nDuplicate Rows:")
print(df[duplicates])

# =========================
# 3. CHECK MISSING VALUES
# =========================
print("\n================ MISSING VALUES ================")

print(df.isnull().sum())

# =========================
# 4. CLEAN TEXT COLUMNS
# =========================
print("\n================ CLEANING TEXT COLUMNS ================")

# Automatically detect text columns
# =========================
# 4. CLEAN TEXT COLUMNS
# =========================
print("\n================ CLEANING TEXT COLUMNS ================")

# Detect text columns safely
text_cols = df.select_dtypes(
    include=['object', 'string', 'str']
).columns

for col in text_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )

print("Text columns cleaned successfully!")
# =========================
# 5. HANDLE MISSING VALUES
# =========================
print("\n================ HANDLING MISSING VALUES ================")

# Replace common empty values with NaN
df.replace(
    ["", " ", "N/A", "NA", "-", "nan"],
    np.nan,
    inplace=True
)

# Drop rows with missing values
df_clean = df.dropna()

print("Dataset Shape After Removing Missing Values:",
      df_clean.shape)

# =========================
# 6. REMOVE DUPLICATES
# =========================
print("\n================ REMOVING DUPLICATES ================")

df_clean = df_clean.drop_duplicates()

print("Remaining Duplicates:",
      df_clean.duplicated().sum())

# =========================
# 7. FINAL DATA CHECK
# =========================
print("\n================ FINAL CLEAN DATA CHECK ================")

print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

print("\nFinal Dataset Shape:",
      df_clean.shape)

print("\nCleaned Dataset Preview:")
print(df_clean.head())

# =========================
# 8. SAVE CLEANED DATASET
# =========================
df_clean.to_csv(
    "iris_cleaned_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")