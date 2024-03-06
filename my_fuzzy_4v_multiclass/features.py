import numpy as np
import pandas as pd


class ProcessArray:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.arrays = [[] for _ in range(num_classes)]

    def set_array(self, class_index, array):
        self.arrays[class_index] = array

    def get_array(self, class_index):
        return self.arrays[class_index]
    
    def concatenate_columns(self):
        return np.column_stack(self.arrays)
    


def feature_selection(vectors_similarities_classes, n_cluster_thesis, num_fuzzy_sets):
    num_classes=len(vectors_similarities_classes)
    delete_items_list=[]

    all_selected_similarities=()

    final_list_centers=[]
    # final_selected_similarities = [None] * len(vectors_similarities_classes)
    # final_selected_similarities=()
    for n_classes, list_vectors_similarities in enumerate(vectors_similarities_classes):
        total_similarities1_per_class=[]

        # print(list_vectors_similarities,"+++++++++\n")
                # selected_similarities_centers1=[]
        for j in range(len(list_vectors_similarities)):
            total_similarities=[]
            selected_similarities_centers1=[]
            list_vectors=list_vectors_similarities[j]

            for m in range(num_classes):
                selected_similarities_centers1=[]

                l=int(n_cluster_thesis[m])
                l1=int(n_cluster_thesis[m+1])
                # print(l,l1)


                # l=n_cluster1
                #print(list_vectors_similarities[j][:l])
                for a in range(len(list_vectors)):
                    for b in range(a + 1, len(list_vectors[l:l1])):
                        if list_vectors[l:l1][a][2]==list_vectors[l:l1][b][2]:
                            mul1= list_vectors[l:l1][a][0]*list_vectors[l:l1][a][1]
                            mul2=list_vectors[l:l1][b][0]*list_vectors[l:l1][b][1]
                            if mul1 >= mul2:
                                index=list_vectors[l:l1].index(list_vectors[l:l1][b])
                                delete_items_list.append(index)
                                list_vectors[l:l1][a][0] = (list_vectors[l:l1][a][0]+list_vectors[l:l1][b][0])/2
                                list_vectors[l:l1][a][1] = (list_vectors[l:l1][a][1]+list_vectors[l:l1][b][1])/2
                            else:
                                index=list_vectors[l:l1].index(list_vectors[l:l1][a])
                                delete_items_list.append(index)
                                list_vectors[l:l1][b][0] = (list_vectors[l:l1][a][0]+list_vectors[l:l1][b][0])/2
                                list_vectors[l:l1][b][1] = (list_vectors[l:l1][a][1]+list_vectors[l:l1][b][1])/2
                        #print(list_vectors_similarities[j][:l][a],list_vectors_similarities[j][:l][b])
                        #print("\n")
                        #print(delete_items_list)
                    
                res = [*set(delete_items_list)]
                final_list_centers.append(res)
                #print(res)
                #list_vectors_similarities1[j][:l].pop(int(np.array(res)))
                delete_vectors = [e for k, e in enumerate(list_vectors[l:l1]) if k not in res] 
                selected_similarities_centers1.append(delete_vectors)
                delete_items_list=[]    

            
            #list_vectors_similarities[j][:l][i].insert(len(list_vectors_similarities[j][:l][i]), 'c1')
                for i in range(len(selected_similarities_centers1)):
                    # print(selected_similarities_centers1[i])
                    for j1 in range(len(selected_similarities_centers1[i])):
                        selected_similarities_centers1[i][j1].insert(len(selected_similarities_centers1[i][j1]),m)
                        weight=(selected_similarities_centers1[i][j1][2]+1)/num_fuzzy_sets
                        selected_similarities_centers1[i][j1].insert(len(selected_similarities_centers1[i][j1]),weight)
                
                # print(selected_similarities_centers1,"??????????????????")
                total_similarities.extend(selected_similarities_centers1)
            #transform the list into 2d
            total_similarities = [item for sublist in total_similarities for item in sublist]
            total_similarities1_per_class.append(total_similarities)
            # print(total_similarities)
        # reduced_similarities_per_sample.append(total_similarities1)
        # print(reduced_similarities_per_sample)
        all_selected_similarities+=(total_similarities1_per_class,)
        

    
    return all_selected_similarities




def create_arrays(array,n_fuzzy_sets):
    num_classes=len(array)
    array_temp=np.arange(num_classes)
    similarities_val=()
    similarities_array = ProcessArray(num_classes) 
    membership_array = ProcessArray(num_classes) 

    membership_val=()
    row = np.repeat(array_temp, [n_fuzzy_sets]*num_classes)

    for n_class, samples in enumerate(array):
        size=(len(samples),n_fuzzy_sets)
        similarities=np.zeros(size) 
        membership=np.zeros(size) 

        for classes in range(num_classes):
            similarities=np.zeros(size) 
            membership=np.zeros(size) 
            for s in range(len(samples)):
                for s1 in range(len(samples[s])):
                    if samples[s][s1][3] == classes:
                            pos=int(samples[s][s1][2])
                            similarities[s,pos]=samples[s][s1][0]
                            membership[s,pos]=samples[s][s1][1]
            similarities_array.set_array(classes, similarities)
            membership_array.set_array(classes, membership)

        similarities_final = similarities_array.concatenate_columns()
        membership_final = membership_array.concatenate_columns()
        membership_val+=(membership_final,)
        similarities_val+=(similarities_final,)
    
    membership_all_classes=np.concatenate(membership_val, axis=0)
    similarities_all_classes=np.concatenate(similarities_val, axis=0)
    num_rows=similarities_all_classes.shape[0]
    array_indexes_classes = np.tile(row, (num_rows, 1))
    


    return similarities_all_classes, membership_all_classes, array_indexes_classes





# def feature_selection(list_vectors_similarities, num_fuzzy_sets, n_cluster1, n_cluster0):
def feature_selection_with_zeros(vectors_similarities_classes, n_cluster_thesis, num_fuzzy_sets):
    num_classes=len(vectors_similarities_classes)
    all_selected_similarities=()
    for n_classes, list_vectors_similarities in enumerate(vectors_similarities_classes):
        total_similarities1_per_class=[]

        for j in range(len(list_vectors_similarities)):
            total_similarities=[]
            total_similarities1=[]

            list_vectors=list_vectors_similarities[j]


            for m in range(num_classes):

                l=int(n_cluster_thesis[m])
                l1=int(n_cluster_thesis[m+1])

                total_similarities= [e for k, e in enumerate(list_vectors[l:l1])]
            
                for i in range(len(total_similarities)):
                    total_similarities[i].insert(len(total_similarities[i]), m)
                    weight=(total_similarities[i][2]+1)/num_fuzzy_sets
                    total_similarities[i].insert(len(total_similarities[i]),weight)
            
                for a in range(len(list_vectors)):
                    for b in range(a + 1, len(list_vectors[l:l1])):
                        # print(list_vectors[l:l1][a],list_vectors[l:l1][b],"++++++++++++++")
                        if list_vectors[l:l1][a][2]==list_vectors[l:l1][b][2]:
                            mul1=list_vectors[l:l1][a][0]*list_vectors[l:l1][a][1]
                            mul2=list_vectors[l:l1][b][0]*list_vectors[l:l1][b][1]
                            if mul1 <= mul2:
                                list_vectors[l:l1][b][4] =  list_vectors[l:l1][b][4] + 0.01
                            else:
                                list_vectors[l:l1][a][4] =  list_vectors[l:l1][a][4] + 0.01



                total_similarities1.extend(total_similarities)
            total_similarities1_per_class.append(total_similarities1)
        all_selected_similarities+=(total_similarities1_per_class,)
    

    
    return all_selected_similarities