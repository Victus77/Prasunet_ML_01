# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())

# Data Preprocessing
# Handling missing values
data = data.dropna()

# Encoding categorical features (if any)
if data['Gender'].dtype == 'object':
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Feature Selection
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the dataset
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(scaled_features)

# Evaluating the clustering with silhouette score
silhouette_avg = silhouette_score(scaled_features, y_kmeans)
print(f'Silhouette Score: {silhouette_avg}')

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=y_kmeans, palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Cluster Profiling
data['Cluster'] = y_kmeans
cluster_profile = data.groupby('Cluster').mean()
print(cluster_profile)

