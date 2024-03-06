import pandas as pd
import numpy as np
from sklearn import metrics






def dummy_with_arrays(similarities, membership, weights, array_classes, n_fuzzy_set):
    
        membership_first_array = membership[:, :n_fuzzy_set]
        membership_second_array = membership[:, n_fuzzy_set:]
        similarities_first_array = similarities[:, :n_fuzzy_set]
        similarities_second_array = similarities[:, n_fuzzy_set:]
        weights_first_array = weights[:, :n_fuzzy_set]
        weights_second_array = weights[:, n_fuzzy_set:]


        print(membership_first_array.shape, similarities_first_array.shape, weights_first_array.shape)

        all_first_arrays=np.multiply(membership_first_array,similarities_first_array,weights_first_array)
        all_second_arrays=np.multiply(membership_second_array,similarities_second_array,weights_second_array)



        total= all_first_arrays[all_first_arrays!=0] - all_second_arrays[all_second_arrays!=0]

        print(total)


        print(all_first_arrays)
        print(all_second_arrays)



def dummy_classification(val_vectors_class1, val_vectors_class0):
    tp=0
    fn=0
    mul_c1=[]
    mul_c0=[]
    scores_c1=[]
    scores_c0=[]
    for i in range(len(val_vectors_class1)):
        for j in range(len(val_vectors_class1[i])):
            # print(val_vectors_class1[i][j],"\n")
            if val_vectors_class1[i][j][3] == 1:
                mul1 = val_vectors_class1[i][j][0]*val_vectors_class1[i][j][1]*val_vectors_class1[i][j][4]
                mul_c1.append(mul1)
            else:
                mul2 = val_vectors_class1[i][j][0]*val_vectors_class1[i][j][1]*val_vectors_class1[i][j][4]
                mul_c0.append(mul2)
            #print(mul_c1)
            
        sum_c0=sum(mul_c0)
        sum_c1=sum(mul_c1)
        #print(sum_c1)
        total=sum_c1-sum_c0
        scores_c1.append(total)
        if total > 0:
            tp=tp + 1
        else:
            fn= fn + 1
        mul_c0=[]
        mul_c1=[]

    fp=0
    tn=0
    for a in range(len(val_vectors_class0)):
        for b in range(len(val_vectors_class0[a])):
            if val_vectors_class0[a][b][3] == 1:
                mul3 = val_vectors_class0[a][b][0]*val_vectors_class0[a][b][1]*val_vectors_class0[a][b][4]
                mul_c1.append(mul3)
            else:
                mul4 = val_vectors_class0[a][b][0]*val_vectors_class0[a][b][1]*val_vectors_class0[a][b][4]
                mul_c0.append(mul4)
        sum_c0=sum(mul_c0)
        sum_c1=sum(mul_c1)

        total1=sum_c1-sum_c0
        scores_c0.append(total1)
        if total1 > 0:
            fp=fp + 1
        else:
            tn= tn + 1
        mul_c0=[]
        mul_c1=[]
    confusion_matrix=[[tp, fp],[fn, tn]]
    accuracy=((tp+tn)/(tp+tn+fp+fn))

    y_true = [1] * len(scores_c1) + [0] * len(scores_c0)
    y_scores = scores_c1 + scores_c0
    auc = metrics.roc_auc_score(y_true, y_scores)
    
    return accuracy, auc, confusion_matrix



def dummy_multiclass_classification(val_vectors_class):
    accuracy=0
    mul_c1=[]
    sum_total=[]
    num_classes=len(val_vectors_class)
    num_all_samples=0
    for n_class, data in enumerate(val_vectors_class):
        num_all_samples+=len(val_vectors_class[n_class])

    for n_classes, data_classes in enumerate(val_vectors_class):
        class_labels = np.delete(np.arange(num_classes), n_classes)


        for i in range(len(data_classes)):
            sum_total=[]
            max_sum=[]
            for classes in range(num_classes):
                mul_c1=[]
                for j in range(len(data_classes[i])):
               
                    # print(val_vectors_class1[i][j],"\n")
                    if data_classes[i][j][3] == classes:
                        mul1 = data_classes[i][j][0]*data_classes[i][j][1]*data_classes[i][j][4]
                        mul_c1.append(mul1)

                
                sum_total.append(sum(mul_c1))
            # max_sum=max(sum_total) dhmioyrgw ayto to max_sum gia oles tis klaseis
            for s in range(len(sum_total)):
                result = 0
                for k in range(len(sum_total)):
                    if s != k:
                        result += (sum_total[s] - sum_total[k])
                max_sum.append(result)
            max_sum1=max(max_sum)
            index_max=max_sum.index(max_sum1)
            if n_classes==index_max:
                accuracy=accuracy+1
    accuracy=accuracy/num_all_samples
            
    
    return accuracy