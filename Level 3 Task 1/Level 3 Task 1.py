# =========================
# LEVEL 3 - TASK 1
# PREDICTIVE MODELING (CLASSIFICATION)
# CUSTOMER CHURN PREDICTION
# =========================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# 2. CREATE OUTPUT FOLDER
# =========================
os.makedirs("outputs", exist_ok=True)

# =========================
# 3. LOAD DATASET
# =========================
df = pd.read_csv("cleaned_churn_80.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =========================
# 4. DATA CLEANING
# =========================
# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# =========================
# 5. HANDLE CATEGORICAL VARIABLES
# =========================
label_encoder = LabelEncoder()

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# =========================
# 6. FEATURE SELECTION
# =========================
# Target column
target_column = "churn"

# Features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# =========================
# 7. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 8. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 9. INITIALIZE MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# =========================
# 10. TRAIN & EVALUATE MODELS
# =========================
results = []

for model_name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # Print results
    print(f"\n{model_name}")
    print("-" * 40)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# =========================
# 11. RESULTS DATAFRAME
# =========================
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

# Save results table
results_df.to_csv("outputs/model_results.csv", index=False)

# =========================
# 12. VISUALIZE MODEL ACCURACY
# =========================
plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results_df
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")

plt.savefig(
    "outputs/model_accuracy_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 13. HYPERPARAMETER TUNING
# RANDOM FOREST + GRID SEARCH
# =========================
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# =========================
# 14. EVALUATE TUNED MODEL
# =========================
best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_predictions)

print("\nTuned Random Forest Accuracy:")
print(best_accuracy)

# =========================
# 15. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 16. FEATURE IMPORTANCE
# =========================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance - Random Forest")

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 17. SAVE BEST MODEL RESULTS
# =========================
feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nProject completed successfully.")
print("All outputs saved in 'outputs/' folder.")