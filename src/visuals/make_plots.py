import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from pandas.plotting import parallel_coordinates
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def create_elbow_chart(range_n_clusters, params, data):
    inertia = []   
    for n_clusters in range_n_clusters: 
        kmeans = KMeans(n_clusters=n_clusters, 
                        **params).fit(data) 
        # Average SS per cluster
        inertia.append(kmeans.inertia_ / n_clusters) 
    inertias = pd.DataFrame({'n_clusters': range_n_clusters, 'inertia': inertia}) 
    ax = inertias.plot(x='n_clusters', y='inertia')
    plt.xlabel('Number of clusters (k)') 
    plt.ylabel('Average Within-Cluster Squared Distances') 
    plt.ylim((0, 1.1 * inertias.inertia.max())) 
    ax.legend().set_visible(False)
    return inertias


def plot_silhouette_scores(range_n_clusters, params, data):
    silhouette_scores = []  
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, 
                        **params).fit(data) 
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)       
    plt.plot(range_n_clusters, silhouette_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel("Avg Silhouette Score")


def create_profile_chart(params, data, ax):
    kmeans = KMeans(**params).fit(data) 
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
    centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
    parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', 
                         linewidth=5, ax=ax)
    plt.xlim(-0.5,len(data.columns)-0.5)
    plt.legend(loc='best')


def visualize_clusters_pca(params, data):
    kmeans = KMeans(**params).fit(data) 
    cluster_assignments = kmeans.fit_predict(data)
    colors = cm.Set1(np.linspace(0, 1, params['n_clusters']))
    cmap = matplotlib.colors.ListedColormap(colors)
    pca = PCA(n_components=2)
    pc1, pc2 = zip(*pca.fit_transform(data))
    plt.scatter(pc1, pc2, c=cluster_assignments.tolist(), cmap=cmap)
    plt.xlabel('PC1')
    plt.ylabel('PC2')