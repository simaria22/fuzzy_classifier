import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import mahalanobis

from sklearn.covariance import MinCovDet

from sklearn.covariance import EllipticEnvelope
import scipy as sp


def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        # Handle the case where one or both vectors have zero length
        similarity = np.nan
    else:
        similarity = dot_product / (norm_x * norm_y)
    
    return similarity

def mahalanobis_distance(x, y, covariance_matrix):
    diff = x - y
    inv_cov_matrix = sp.linalg.inv(covariance_matrix)
    left_term = np.dot(diff, inv_cov_matrix)
    mahal = np.dot(left_term, diff.T)
    return mahal


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


def similarity_measure_vector1(scaled_df, all_centers):
    total_list_similarities1 = []
    new_array = np.empty((len(scaled_df),), dtype=np.ndarray)
    # total_list_similarities1 = ()
    shape_centers=0
    shape_features=scaled_df[0].shape[1]

    for i, data_centers in enumerate(all_centers):
        shape_centers+=all_centers[i].shape[0]


    for k, data in enumerate(all_centers):
        for c in range(len(all_centers[k])):
            all_centers[k][c]=all_centers[k][c].reshape(1,shape_features)


    centers = np.concatenate(all_centers,axis=0)
    for j, classes in enumerate(scaled_df):
        total_list_similarities=[]
        # shape_centers=all_centers[j].shape[0]
        for i,row in enumerate(classes):
            distances=[]
            total_dist=0

            # distances = calculate_mahalanobis_distances(classes, centers, cov_matrix)


            for a, data in enumerate(all_centers):
                for center in range(len(all_centers[a])):

                    temp=row - all_centers[a][center]
                    sum_sq = np.dot(temp.T, temp)
                    arr=np.sqrt(sum_sq)
                    # arr=manhattan_distance(row, all_centers[a][center])

                    total_dist+=arr

                    
            for v_center, data in enumerate(all_centers):  
                for cen in range(len(all_centers[v_center])):

                    # s1=manhattan_distance(row, all_centers[v_center][cen])
                    # cos_sim=cosine_similarity(row, all_centers[v_center][cen])
                    temp = row - all_centers[v_center][cen]
                    sum_sq = np.dot(temp.T, temp)
                    s1 = np.sqrt(sum_sq)

                    s = 1 - (s1 / total_dist)
                    distances.append(s)
                        
            distances=np.array(distances)
            distances=distances.reshape(1,shape_centers)
            #print("The similarity measure of data", index, "as vector of similarities is:" ,distances)
            total_list_similarities.append(distances)
        total_list_similarities1=np.concatenate(total_list_similarities, axis=0)
        new_array[j]=total_list_similarities1



    return new_array





def similarity_measure_vector2(scaled_df, all_centers):
    total_list_similarities1 = []
    shape_centers = 0
    shape_features = scaled_df[0].shape[1]

    for i, data_centers in enumerate(all_centers):
        shape_centers += all_centers[i].shape[0]

    for k, data in enumerate(all_centers):
        for c in range(len(all_centers[k])):
            all_centers[k][c] = all_centers[k][c].reshape(1, shape_features)
    all_data = np.concatenate(scaled_df,axis=0)
    cov_matrix = np.cov(all_data.T)  + np.eye(all_data.shape[1]) * 1e-6

    for j, classes in enumerate(scaled_df):
        # np.set_printoptions(precision=4, suppress=True)

        total_list_similarities = []
        for i, data_set in enumerate(classes):
            distances = []
            total_dist = 0


            # Calculate the total Mahalanobis distance for normalization
            for a, data in enumerate(all_centers):
                for center in range(len(all_centers[a])):    
                     # Transpose the data for covariance calculation
                    arr = mahalanobis_distance(data_set,all_centers[a][center], cov_matrix)
                    # arr= np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff.T))
                    total_dist += arr

            # Calculate the similarity measure
            for v_center, data in enumerate(all_centers):  
                for cen in range(len(all_centers[v_center])):
                    # s1 = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff.T))
                    s1 = mahalanobis_distance(data_set, all_centers[v_center][cen], cov_matrix)

                    
                    s = 1 - (s1 / total_dist)
                    distances.append(s)

            distances = np.array(distances)
            distances = distances.reshape(1, len(distances))
            total_list_similarities.append(distances)

        total_list_similarities1.append(np.concatenate(total_list_similarities, axis=0))

    return np.array(total_list_similarities1)






def connect_class_with_class_centers(similarities_train, n_clusters):
    class_centers = np.empty(len(similarities_train), dtype=object)
    n_cluster_thesis=np.empty((len(n_clusters)+1))
    n_cluster_thesis[0]=0
    counter=1
    # n_cluster_thesis=[0,n_clusters[0],n_clusters[0]+n_clusters[1]]
    # [0,2,5]
    # [2,3]

    for i in n_clusters:
        n_cluster_thesis[counter]=i+n_cluster_thesis[counter-1]
        counter+=1

    # number_of_classes=similarities_train.shape[0]
    # l1=len(similarities_train[0][0])//number_of_classes

    for num_classes, class_sim in enumerate(similarities_train):
        l1=int(n_cluster_thesis[num_classes])
        l2=int(n_cluster_thesis[num_classes+1])
        class_centers[num_classes]=class_sim[:,l1:l2]

    return class_centers, n_cluster_thesis

