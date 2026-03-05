import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data # Features
# Standardize the dataset for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Define different values of K for testing
K_values = range(1, 11) # Testing K from 1 to 10
inertia_values = [] # Store sum of squared Euclidean distances (inertia)
for K in K_values:
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_) # Inertia is the sum of squared distances
# Plot the inertia values vs. K
plt.figure(figsize=(8, 5))
plt.plot(K_values, inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('K-Means Performance: Inertia vs. K')
plt.xticks(K_values)
plt.grid()
plt.show()