import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# 0. CREATE FIGURES FOLDER
# =========================
os.makedirs("figures", exist_ok=True)

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("Stock_Prices.csv")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
)

print("\n================ DATA PREVIEW ================")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:", df.shape)

# =========================
# 3. HANDLE DATE COLUMN
# =========================
date_col = None

for col in df.columns:
    if "date" in col:
        date_col = col
        break

if date_col is None:
    raise ValueError("No date column found in dataset!")

# Convert date column
df[date_col] = pd.to_datetime(df[date_col])

# Sort by date
df = df.sort_values(date_col)

# Set date as index
df.set_index(date_col, inplace=True)

# =========================
# 4. IDENTIFY PRICE COLUMN
# =========================
price_col = None

for col in df.columns:
    if col in ["close", "price", "adj close", "adj_close"]:
        price_col = col
        break

# Fallback if not found
if price_col is None:
    price_col = df.select_dtypes(include=np.number).columns[0]

print("\nUsing Price Column:", price_col)

# =========================
# 5. HANDLE MISSING VALUES
# =========================
df[price_col] = df[price_col].ffill()

# =========================
# 6. FILTER SINGLE STOCK
# =========================
# Using AAPL for cleaner visualization

if "symbol" in df.columns:
    df = df[df["symbol"] == "AAPL"]

print("\nFiltered Dataset Shape:", df.shape)

# =========================
# 7. PLOT ORIGINAL TIME SERIES
# =========================
plt.figure(figsize=(12, 5))

plt.plot(df[price_col])

plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")

# Save Figure
plt.savefig(
    "figures/time_series_plot.png",
    dpi=300,
    bbox_inches="tight"
)

# Show Plot
plt.show()

# Close Plot
plt.close()

# =========================
# 8. MOVING AVERAGE
# =========================
df["moving_avg"] = (
    df[price_col]
    .rolling(window=10)
    .mean()
)

plt.figure(figsize=(12, 5))

plt.plot(
    df[price_col],
    label="Original"
)

plt.plot(
    df["moving_avg"],
    label="Moving Average",
    color="red"
)

plt.title("Moving Average Smoothing")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

# Save Figure
plt.savefig(
    "figures/moving_average_plot.png",
    dpi=300,
    bbox_inches="tight"
)

# Show Plot
plt.show()

# Close Plot
plt.close()

# =========================
# 9. TIME SERIES DECOMPOSITION
# =========================
# Use smaller sample for faster processing

series = df[price_col].dropna()

series = series.iloc[:365]

decomposition = seasonal_decompose(
    series,
    model="additive",
    period=30
)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# =========================
# 10. PLOT DECOMPOSITION
# =========================
plt.figure(figsize=(12, 8))

# Original
plt.subplot(411)
plt.plot(series)
plt.title("Original Series")

# Trend
plt.subplot(412)
plt.plot(trend)
plt.title("Trend")

# Seasonal
plt.subplot(413)
plt.plot(seasonal)
plt.title("Seasonality")

# Residual
plt.subplot(414)
plt.plot(residual)
plt.title("Residuals")

plt.tight_layout()

# Save Figure
plt.savefig(
    "figures/time_series_decomposition.png",
    dpi=300,
    bbox_inches="tight"
)

# Show Plot
plt.show()

# Close Plot
plt.close()

# =========================
# 11. SAVE CLEAN DATASET
# =========================
df.to_csv(
    "cleaned_stock_data.csv"
)

# =========================
# 12. SUMMARY INSIGHTS
# =========================
print("\n================ ANALYSIS COMPLETE ================")

print("\nInsights:")
print("- Trend shows long-term movement in stock prices.")
print("- Seasonality reveals repeating patterns.")
print("- Residuals represent random market fluctuations.")
print("- Moving averages smooth short-term volatility.")

print("\nSaved Files:")
print("1. cleaned_stock_data.csv")
print("2. figures/time_series_plot.png")
print("3. figures/moving_average_plot.png")
print("4. figures/time_series_decomposition.png")

print("\nLevel 2 Task 2 completed successfully!")