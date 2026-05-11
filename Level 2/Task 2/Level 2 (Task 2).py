import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("Stock_Prices.csv")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.lower().str.strip()

print("\n================ DATA PREVIEW ================")
print(df.head())
print(df.info())

# =========================
# 3. HANDLE DATE COLUMN SAFELY
# =========================
# Try to detect date column
date_col = None
for col in df.columns:
    if "date" in col:
        date_col = col
        break

if date_col is None:
    raise ValueError("No date column found in dataset")

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)

# =========================
# 4. IDENTIFY PRICE COLUMN
# =========================
price_col = None
for col in df.columns:
    if col in ["close", "price", "adj close", "adj_close"]:
        price_col = col
        break

if price_col is None:
    # fallback: assume first numeric column
    price_col = df.select_dtypes(include=np.number).columns[0]

print("\nUsing price column:", price_col)

# =========================
# 5. PLOT ORIGINAL TIME SERIES
# =========================
plt.figure(figsize=(12,5))
plt.plot(df[price_col])
plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# =========================
# 6. MOVING AVERAGE (SMOOTHING)
# =========================
df["moving_avg"] = df[price_col].rolling(window=10).mean()

plt.figure(figsize=(12,5))
plt.plot(df[price_col], label="Original")
plt.plot(df["moving_avg"], label="Moving Average", color="red")
plt.title("Moving Average Smoothing")
plt.legend()
plt.show()

# =========================
# 7. HANDLE MISSING VALUES (IF ANY)
# =========================
df[price_col] = df[price_col].ffill()

# =========================
# 8. TIME SERIES DECOMPOSITION
# =========================
# Choose period safely (adjust if needed)
period = 30 if len(df) > 60 else 12

decomposition = seasonal_decompose(df[price_col], model="additive", period=period)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# =========================
# 9. PLOT DECOMPOSITION
# =========================
plt.figure(figsize=(12,8))

plt.subplot(411)
plt.plot(df[price_col])
plt.title("Original Series")

plt.subplot(412)
plt.plot(trend)
plt.title("Trend")

plt.subplot(413)
plt.plot(seasonal)
plt.title("Seasonality")

plt.subplot(414)
plt.plot(residual)
plt.title("Residuals")

plt.tight_layout()


# SAVE FIGURE
plt.savefig("time_series_decomposition.png")

plt.show()
# =========================
# 10. SUMMARY INSIGHTS
# =========================
print("\n================ INSIGHTS ================")

print("Trend shows overall long-term movement of the stock price.")
print("Seasonality shows repeating patterns in the data.")
print("Residuals represent random noise or market volatility.")

# =========================
# 11. SAVE CLEAN DATA
# =========================
df.to_csv("cleaned_stock_data.csv")

print("\nCleaned dataset saved successfully!")

print("\nTask 2, level 2 completed successfully!")