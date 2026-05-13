# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================
# 2. CREATE OUTPUT FOLDER
# =========================
os.makedirs("outputs", exist_ok=True)

# =========================
# 3. LOAD DATASET (FIXED)
# =========================
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

# =========================
# 4. SELECT FEATURES
# =========================
X = df.drop('species', axis=1)

# =========================
# 5. STANDARDIZE DATA
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 6. ELBOW METHOD
# =========================
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal K (Iris Dataset)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.savefig("outputs/elbow_method.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 7. APPLY K-MEANS (K=3)
# =========================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters

# =========================
# 8. PCA FOR 2D VISUALIZATION
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# =========================
# 9. CLUSTER VISUALIZATION
# =========================
plt.figure(figsize=(8,6))

sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    data=df,
    palette='viridis',
    s=80
)

centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=200,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title("K-Means Clustering (Iris Dataset)")
plt.legend()

plt.savefig("outputs/kmeans_clusters.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 10. SAVE FINAL DATASET
# =========================
df.to_csv("outputs/iris_clustered_data.csv", index=False)

print("All outputs saved in 'outputs/' folder successfully.")