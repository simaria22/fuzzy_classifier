import pandas as pd
import numpy as np
import skfuzzy as fuzz
from itertools import chain
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fcmeans import FCM
from sklearn.mixture import GaussianMixture


def fuzzification(similarities, n_cluster):
    fuzzy_sets=[]
    sets_centers=[]
    all_classes_fuzzy_sets = ()

  
    x = np.arange(0,1,0.00001)
    for c, class_similarities in enumerate(similarities):
        class_similarities_transposed=class_similarities.T

        for z,col in enumerate(class_similarities_transposed):
            fuzzy_sets.append([])

            # kmeans = KMeans(init="k-means++",n_clusters=n_cluster, n_init=1, max_iter=300,random_state=None)
            gmm = GaussianMixture(n_components=n_cluster, n_init=1, max_iter=300, random_state=None)

            col2=col.reshape(-1,1)
            
            # kmeans.fit(col.reshape(-1,1))
            gmm.fit(col.reshape(-1,1))
            # list_centers=kmeans.cluster_centers_
            list_centers=gmm.means_
            list_centers=sorted(list_centers)
            list_centers=np.insert(list_centers, n_cluster, 0)
            list_centers=np.insert(list_centers, n_cluster, 1)
            #print(list_centers)
            # print("break point 8")
            for i in range(n_cluster-1):        
                centers=[list_centers[i-1], list_centers[i], list_centers[i+1]]
                fuzzy_sets[z].append(centers)
                # print("break point 11")
                # plt.plot(x,mfx)
            centers=[list_centers[n_cluster-2], list_centers[n_cluster-1], list_centers[n_cluster],list_centers[n_cluster]]
            # print(fuzzy_sets)
            fuzzy_sets[z].append(centers)
            # print("break point 12")
            # plt.plot(x,mfx)
            # plt.title('Membership functions of clusters centers')
            # plt.show()
        # print(fuzzy_sets)
        all_classes_fuzzy_sets+=(fuzzy_sets,)
        fuzzy_sets=[]
        # print("break point 14")

    return all_classes_fuzzy_sets


def gaussian(x, m, sigma):
    return np.exp(-((x - m) ** 2) / (sigma ** 2))



def vectorization_similarities(class_similarities, fuzzy_sets):
    total_fuzzy=np.concatenate(fuzzy_sets, axis=0)
    # total_fuzzy_sets=fuzzy_sets
    x = np.arange(0,1,0.00001)
    array_membership=[]
    dataframes={}
    list_membership_values=[]
    list_degree_values=[]
    df_list=[]
    dic={}
    # final_list = [None] * len(class_similarities)
    final_list=()


    for index, class_normalized_similarities in enumerate(class_similarities):
        df_list=[]
        class1_similarities=class_normalized_similarities.T
        # print(class1_similarities)
        # print(range(len(total_fuzzy_sets)))
        values1=[]
        for c in range(class1_similarities.shape[0]):
                values=[f"{c+1}"]
                values1.append(values)
        values1=np.array(values1)
        values1=values1[:,0]
        # print(class1_similarities.shape[0], len(total_fuzzy))
        for i,col,names in zip(range(len(total_fuzzy)), class1_similarities, values1):

            # print(total_fuzzy_sets[i])
            for k in range(len(col)):
                col_similarity=col[k]
                for j in range(len(total_fuzzy[i])):
                    if(col_similarity<= total_fuzzy[i][j][0] ):
                        mem = 0
                    elif(col_similarity<= total_fuzzy[i][j][1] and col_similarity > total_fuzzy[i][j][0]):
                        # mem= gaussian(col_similarity, total_fuzzy[i][j][1], total_fuzzy[i][j][0])
                        mem = (col_similarity-total_fuzzy[i][j][0])/(total_fuzzy[i][j][1]-total_fuzzy[i][j][0])
                    elif(col_similarity <= total_fuzzy[i][j][2] and col_similarity> total_fuzzy[i][j][1]):
                        # mem= gaussian(col_similarity,total_fuzzy[i][j][2], total_fuzzy[i][j][1])
                        mem = (total_fuzzy[i][j][2]-col_similarity)/(total_fuzzy[i][j][2]-total_fuzzy[i][j][1])
                    elif(col_similarity > total_fuzzy[i][j][2]):
                        if len(total_fuzzy[i][j]) == 4:
                            mem = 1
                        else:
                            mem=0
                        # trap=fuzz.trapmf(x,total_fuzzy[i][j])
                        # mem2=fuzz.interp_membership(x, trap, col_similarity)
                    
                    array_membership.append(mem)
                max_membership_value=max(array_membership)
                index = array_membership.index(max_membership_value)
                array_membership=[]

                list_membership_values.append(max_membership_value)
                list_degree_values.append(index)
            #print(list_membership_values)
            dict = {f'similarities{names}': col, f'membership_values{names}': list_membership_values, f'degree{names}': list_degree_values} 
            df = pd.DataFrame(dict)
            dataframes[names]=df
            listt=df.values
            df_list.append(listt)
            list_membership_values=[]
            list_degree_values=[]

            lst=[list(chain.from_iterable([i])) for i in zip(*df_list)]
            lst = np.array(lst).tolist()
        final_list+=(lst,)
    
    return final_list