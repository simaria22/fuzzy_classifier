import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import sys
from fcmeans import FCM
from sklearn.mixture import GaussianMixture



def separate_classes2(array, class_col):
    unique_labels = np.unique(class_col)
    result_arrays = []
    for label in unique_labels:
        mask = class_col == label
        result_arrays.append(array[mask.flatten(), :])
    return result_arrays


def my_clustering(data, n_cluster, estimator='gmm'):
    medoids = np.empty(data.shape[0], dtype=object)
    centers = np.empty(data.shape[0], dtype=object)
    if estimator == 'kmeans':
        for i, df in enumerate(data):

            try:
                if estimator == 'kmeans':
                    #data2=data[data_classes==i]
                    kmeans = KMeans(init="k-means++",n_clusters=n_cluster[i], n_init=1, max_iter=300, random_state=None)
                    kmeans.fit(df)
                    list_centers=kmeans.cluster_centers_
                    data = df
                    centers[i]=list_centers

                medoid_indices = []
                for j in range(n_cluster[i]):
                    distances = np.linalg.norm(data - list_centers[j], axis=1)
                    medoid_index = np.argmin(distances)
                    medoid_indices.append(medoid_index)
                medoids[i] = data[medoid_indices]
            except ValueError as e:
                if 'n_samples' in str(e) and 'should be >= n_clusters' in str(e):
                    print("Error: Number of samples is less than the number of clusters")
                    sys.exit()
                else:
                    raise e
            

    elif estimator=='gmm':
        for i, df in enumerate(data):

            gmm = GaussianMixture(n_components=n_cluster[i], n_init=1, max_iter=300, random_state=None)
            gmm.fit(df)
            list_centers = gmm.means_
            centers[i]=list_centers

                    
    return medoids,centers

    