import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def optimise_k_means(data, max_h):
    means = []
    ineriatas = []
    for k in range(1, max_h):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        ineriatas.append(kmeans.inertia_)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(means, ineriatas, 'o-')
    plt.xlabel('No of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

s = """
    1. Elbow method 
    2. Đưa k tâm dựa theo Elbow method
    3. Trực quan hóa sau khi đã phân cụm
"""
df = pd.read_csv('winequality-red.csv')
df.drop(columns = ['quality','fixed acidity','volatile acidity'],inplace = True)
X = df.values
X = StandardScaler().fit_transform(X)

ok = 0

print(s)
# n = int(input())
optimise_k_means(X,30)

print("Nhap k cum: ",end = ' ')
k = int(input())
clr = KMeans(n_clusters= k , init='k-means++', random_state=6)
clr.fit(X)
print(clr.cluster_centers_)

X = PCA(2).fit_transform(X)
df = PCA(2).fit_transform(df)
clr = KMeans(n_clusters=10, init='k-means++', random_state=6)
clr.fit(df)
labels = clr.predict(df)
centroid_labels = clr.predict(clr.cluster_centers_)
predicted_df = pd.DataFrame(data=df, columns=['PCA1', 'PCA2'])
predicted_df['Cluster'] = labels
centroid_df = pd.DataFrame(data=clr.cluster_centers_, columns=['PCA1', 'PCA2'])
centroid_df['Cluster'] = centroid_labels
plt.scatter(predicted_df['PCA1'], predicted_df['PCA2'], c=labels, alpha=0.8)
plt.scatter(centroid_df['PCA1'], centroid_df['PCA2'], marker='+', s=100, c='red')
plt.show()



