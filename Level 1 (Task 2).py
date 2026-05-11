# =====================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# DATASET: IRIS DATASET
# =====================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2. DISPLAY SETTINGS
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Set plot style
sns.set_style("whitegrid")

# =========================
# 3. LOAD CLEANED DATASET
# =========================
df = pd.read_csv("iris_cleaned_dataset.csv")

# =========================
# 4. DATA INSPECTION
# =========================
print("\n================ DATA INSPECTION ================")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

# =========================
# 5. SUMMARY STATISTICS
# =========================
print("\n================ SUMMARY STATISTICS ================")

print("\nDescriptive Statistics:")
print(df.describe())

# =========================
# 6. MEAN
# =========================
print("\n================ MEAN ================")

mean_values = df.mean(numeric_only=True)

print(mean_values)

# =========================
# 7. MEDIAN
# =========================
print("\n================ MEDIAN ================")

median_values = df.median(numeric_only=True)

print(median_values)

# =========================
# 8. MODE
# =========================
print("\n================ MODE ================")

mode_values = df.mode()

print(mode_values)

# =========================
# 9. STANDARD DEVIATION
# =========================
print("\n================ STANDARD DEVIATION ================")

std_values = df.std(numeric_only=True)

print(std_values)

# =========================
# 10. HISTOGRAMS
# =========================
print("\n================ HISTOGRAMS ================")

numeric_columns = df.select_dtypes(
    include=['float64', 'int64']
).columns

for col in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[col],
        bins=15,
        kde=True
    )

    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Save histogram
    plt.savefig(
        f"{col}_histogram.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

# =========================
# 11. BOXPLOTS
# =========================
print("\n================ BOXPLOTS ================")

for col in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    # Save boxplot
    plt.savefig(
        f"{col}_boxplot.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

# =========================
# 12. SCATTER PLOTS
# =========================
print("\n================ SCATTER PLOTS ================")

# Scatter Plot 1
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species"
)

plt.title("Petal Length vs Petal Width")

# Save scatter plot
plt.savefig(
    "petal_scatterplot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# Scatter Plot 2
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue="species"
)

plt.title("Sepal Length vs Sepal Width")

# Save scatter plot
plt.savefig(
    "sepal_scatterplot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 13. CORRELATION ANALYSIS
# =========================
print("\n================ CORRELATION MATRIX ================")

correlation_matrix = df.corr(
    numeric_only=True
)

print(correlation_matrix)

# =========================
# 14. CORRELATION HEATMAP
# =========================
print("\n================ CORRELATION HEATMAP ================")

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

# Save heatmap
plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 15. PAIRPLOT
# =========================
print("\n================ PAIRPLOT ================")

pairplot = sns.pairplot(
    df,
    hue="species"
)

# Save pairplot
pairplot.savefig(
    "pairplot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 16. SPECIES DISTRIBUTION
# =========================
print("\n================ SPECIES DISTRIBUTION ================")

print(df["species"].value_counts())

plt.figure(figsize=(6, 5))

sns.countplot(
    x="species",
    data=df
)

plt.title("Species Distribution")

# Save countplot
plt.savefig(
    "species_distribution.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 17. FINAL OBSERVATIONS
# =========================
print("\n================ FINAL OBSERVATIONS ================")

print("""
1. The dataset contains both numerical and categorical features.

2. Duplicate rows were successfully removed during data cleaning.

3. No missing values were found in the dataset.

4. Petal length and petal width show strong positive relationships.

5. Some numerical features are highly correlated.

6. Setosa species appears clearly separated from the other species.

7. Histograms reveal the distribution of numerical variables.

8. Boxplots help identify spread and possible outliers.

9. Scatter plots reveal patterns and clustering among species.

10. The heatmap provides a clear view of feature correlations.
""")

print("\nEDA COMPLETED SUCCESSFULLY!")

# =========================
# 18. SAVE FINAL DATASET
# =========================
df.to_csv(
    "iris_eda_dataset.csv",
    index=False
)

print("\nEDA dataset saved successfully!")
