# Level 2 – Task 2: Time Series Analysis

## 📌 Project Overview

This project focuses on performing **Time Series Analysis** on stock market data using Python. The analysis was carried out to identify trends, seasonality, and random fluctuations in stock prices over time.

The project includes:

* Data cleaning and preprocessing
* Time series visualization
* Moving average smoothing
* Seasonal decomposition
* Trend and residual analysis

---

# 🛠️ Tools & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Statsmodels

---

# 📂 Dataset Information

Dataset used:
Stock Price Dataset

The dataset contains:

* Stock symbols
* Trading dates
* Opening prices
* Closing prices
* Highest prices
* Lowest prices
* Trading volumes

---

# 📊 Project Objectives

The objectives of this task were to:

* Analyze stock price movements over time
* Detect trends and seasonality
* Perform moving average smoothing
* Decompose the time series into:

  * Trend
  * Seasonal
  * Residual components
* Visualize the results using plots

---

# 🧹 Data Cleaning & Preparation

The following preprocessing steps were performed:

1. Loaded the dataset using pandas
2. Standardized column names
3. Converted the date column to datetime format
4. Sorted the dataset chronologically
5. Set the date column as the index
6. Handled missing values using forward filling
7. Selected the closing price column for analysis
8. Filtered a single stock symbol (`AAPL`) for cleaner visualization

---

# 📈 Time Series Visualization

A line chart was created to visualize stock price movement over time.

### What was observed:

* Continuous fluctuations in stock prices
* Long-term upward and downward movements
* Market volatility over time

Saved Figure:

```text id="m4f22y"
time_series_plot.png
```

---

# 📉 Moving Average Smoothing

A 10-day moving average was calculated to smooth short-term fluctuations in stock prices.

### Purpose:

* Reduce noise in the data
* Highlight the overall trend more clearly

Saved Figure:

```text id="e6c2lt"
moving_average_plot.png
```

---

# 🔍 Time Series Decomposition

The stock price series was decomposed into four major components using `statsmodels`.

## Components:

### 1. Original Series

The raw stock price data.

### 2. Trend

Shows the long-term movement of stock prices.

### 3. Seasonality

Displays repeating patterns or cycles within the data.

### 4. Residuals

Represents random fluctuations and market noise.

Saved Figure:

```text id="d78m8u"
time_series_decomposition.png
```

---

# 📌 Key Findings

* Stock prices showed noticeable fluctuations over time.
* Moving averages helped smooth volatility and highlight trends.
* Seasonal decomposition revealed repeating market patterns.
* Residual analysis captured random market behavior and noise.

---

# 📁 Output Files

The following files were generated during the project:

```text id="7qax6q"
cleaned_stock_data.csv
time_series_plot.png
moving_average_plot.png
time_series_decomposition.png
```

---

# 🚀 Conclusion

This project successfully demonstrated how time series analysis techniques can be applied to stock market data using Python.

The analysis provided insights into:

* Long-term trends
* Seasonal patterns
* Market volatility
* Data smoothing techniques

This task also demonstrated practical usage of:

* Pandas for preprocessing
* Matplotlib for visualization
* Statsmodels for decomposition analysis

---

# 💻 How to Run the Project

## Install Required Libraries

```bash id="2mjlwm"
pip install pandas numpy matplotlib statsmodels
```

## Run the Python Script

```bash id="x49yr2"
python "Level 2 (Task 2).py"
```

---

# 📷 Generated Visualizations

The project automatically saves:

* Original time series plot
* Moving average smoothing plot
* Time series decomposition plot

These visualizations can be included in reports or presentations.

---

# 👨‍💻 Author

Iyiola Olofin
