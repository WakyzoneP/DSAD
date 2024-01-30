import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

alchool_df = pd.read_csv("alchool.csv")
continents_df = pd.read_csv("continents.csv")

merged_df = pd.merge(continents_df, alchool_df, on="continent")

numeric_cols = merged_df.columns[3:].to_list()

# Calcul ierarhie (matricea ierarhie)
def standardize(table):
    mean = np.mean(table, axis=0)
    std = np.std(table, axis=0)
    return (table - mean) / std

X = merged_df[numeric_cols].values
X_std = standardize(X)

link = linkage(X_std, method='ward', metric='euclidean')
print(link)

# Calcul partiție optimală (repartizarea instanțelor în clusteri) prin metoda Elbow pe baza diferențelor dintre distanțele de agregare (Nu Elbow KMeans!)
last = link[-10:, 2]
acceleration = np.diff(last, 2)
k = acceleration.argmax() + 2
print("Numarul de clustere optim este: ", k)

# Calcul partiție oarecare (cu un număr prestabilit de clusteri - numărul de clusteri se inițializează prin cod)
clusters = fcluster(link, k, criterion='maxclust') #  k -- numărul de clusteri
print(clusters)

# Calcul indecși Silhouette la nivel de partiție
silhouette_avg = silhouette_score(X_std, clusters)
print("Silhouette Score: ", silhouette_avg)

# Calcul indecși Silhouette la nivel de instanță
sample_silhouette_values = silhouette_samples(X_std, clusters)
print("Sample Silhouette Values: ", sample_silhouette_values)

# Trasare plot dendrogramă cu evidențierea partiției prin culoare (optimală și partiție-k)
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distanță")
dendrogram(link, leaf_rotation=90., leaf_font_size=8., show_leaf_counts=True, truncate_mode='level', p=int(k), show_contracted=True), # p -- numărul de clusteri
plt.axhline(y=link[-10:, 2].max(), c='k')
plt.axhline(y=link[-k, 2], c='k')
plt.savefig("dendrogram.png")

# Trasare plot Silhouette partiție (optimală și partiție-k)
plt.figure(figsize=(10, 7))
plt.title("Silhouette")
plt.xlabel("Silhouette")
plt.ylabel("Index")
y_lower = 10
for i in range(k):
    ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
plt.axvline(x=silhouette_avg, c='k')
plt.savefig("silhouette.png")

# Trasare histograme clusteri pentru fiecare variabilă observată (partiție optimală și partiție-k)
for col in numeric_cols:
    plt.figure(figsize=(10, 7))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Index")
    for i in range(k):
        plt.hist(X_std[clusters == i, numeric_cols.index(col)], alpha=0.7)
    plt.savefig(col + ".png")

# Trasare plot partiție în axe principale (partiție optimală și partiție-k) ??? 
plt.figure(figsize=(10, 7))
plt.title("PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
for i in range(k):
    plt.scatter(X_std[clusters == i, 0], X_std[clusters == i, 1], alpha=0.7)
plt.savefig("pca.png")