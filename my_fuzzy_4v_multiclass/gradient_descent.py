import imp
from random import sample
import math
from unittest import result
import numpy as np
import itertools
from collections import Counter
from itertools import chain
import extract_rules
import sys
import re
from sklearn import metrics
import extract_rules
import features

 


# def compare(s, t):
#     return Counter(chain(*s)) == Counter(chain(*t))

 


def return_equal_sample(rule, samples, class_vectors):
    l=[]
    l1=[]
    s=[]
    count=0
    index_sample=[]
    num_classes=len(np.unique(class_vectors).astype(int))

 

    for i in range(len(samples)):

 

        label_class_vector = int(class_vectors[i].item())
        for n_classes in range(num_classes-1):
            l=[]
            l1=[]
            for j in range(len(samples[i])):
                class_labels = np.delete(np.arange(num_classes), label_class_vector) 

                if samples[i][j][3] == label_class_vector or samples[i][j][3]==class_labels[n_classes]:
                    selected_indexes=samples[i][j][2:4]
                    l.append(selected_indexes)
            # sample_equal1.append(l)
        # sample_equal.append(sample_equal1)
        # print(sample_equal1,"....................\n")

 

            for k in range(len(rule)):
                if k%2!=0:
                    l1.append(rule[k][0:2])
            if extract_rules.compare(l,l1)==True:
                index_sample1=samples.index(samples[i])
                count+=1
                s.append(samples[i])
                index_sample.append(index_sample1)
        l=[]
        l1=[]
    return s, count, index_sample

 

    

 

 

def error_calc(similarities, membership, rules_signs, rules_weights, class_per_rule, count, class_vectors, array_indexes):



 

    sum_total=[]
    errors=[]
 


    for index, (sample_s,sample_m) in enumerate(zip(similarities,membership)):
        get_index=array_indexes[index]
        result_mul=np.multiply(np.multiply(np.multiply(sample_s, sample_m), rules_weights),rules_signs)
        total=sum(result_mul)
        sum_total.append(total)

        if class_vectors[get_index] == class_per_rule:
            target=1
            if total>1:
                total=1
                error=total-target
            else:
                error=total-target
        else:
            target=0
            if total<0:
                total=0
                error=total-target
            else:
                error=total-target

 

        errors.append(error)
    # print(errors)
    # if count == 0:
    #     count=1
        
    MSE = sum([(2 * error) for error in errors]) / count
    return MSE

 

 

 

 


def update_weights_gd(rules_base, samples, learning_rate, class_vectors, num_iters, n_fuzzy_sets):

 

    all_rules = sum(rules_base, [])
    all_samples_val_set=sum(samples,[])

 



    similarities_val, membership_val, array_indexes_classes = features.create_arrays(samples,n_fuzzy_sets)
    class_per_rule,classes_compare=extract_rules.create_arrays_rules(rules_base)
    rules_signs, rules_weights=extract_rules.create_array_signs(rules_base,class_per_rule,classes_compare, n_fuzzy_sets)

 

    for a in range(len(all_rules)):

        iter=1
        mask = np.isin(array_indexes_classes, classes_compare[a])
        similarities_c = similarities_val[:, mask.any(axis=0)]
        # membership_c = membership_val[np.isin(array_indexes_classes, classes_compare[a])]


# Extract the values using fancy indexing
        # similarities_c = similarities_val[np.arange(array_indexes_classes.shape[0])[:, np.newaxis], array_indexes_classes == classes_compare[a][0] | array_indexes_classes == classes_compare[a][1]]
        membership_c = membership_val[:, mask.any(axis=0)]


        s, count, indexes = return_equal_sample(all_rules[a], all_samples_val_set, class_vectors)
        similarities_temp=similarities_c[indexes]
        membership_temp=membership_c[indexes]

 

        while(iter<=num_iters):
            if iter>2:
                if abs(error)<0.0001 or math.isnan(error):
                    break
            error=error_calc(similarities_temp, membership_temp, rules_signs[a], rules_weights[a], class_per_rule[a], count, class_vectors, indexes) 
            if math.isnan(error) or math.isinf(error):
                break
            for similarity, membership in zip(similarities_temp,membership_temp):
                mul= np.multiply(np.multiply(similarity,membership), rules_signs[a])
                tmp_weight=[]
                for index,weight in enumerate(rules_weights[a]):
                    if weight != 0:
                        new_weight=weight - (error * mul[index] * learning_rate)
                        rules_weights[a][index]=new_weight
                        tmp_weight.append(new_weight)

 

                for i in range(len(all_rules[a])):
                    if i%2!=0:
                        # for index,weight in enumerate(rules_weights[a]):
                    # new_weight_corrected=all_rules[a][i][2] - (error * mul[i] * learning_rate)
                        all_rules[a][i][2]=tmp_weight[i//2]
            iter=iter+1
    #print(r2)
    return all_rules, error

 

 


def activate_rules(sample, rules_base, number_of_classes):
    samples=[]
    base=[]
    samples_list=[]
    activate_rules=[]
    activate_rule=[]
    final_rules=[]
    count=0

 

    for n in range(number_of_classes):
        samples_list=[]
        for n1 in range(n+1, number_of_classes):
            samples_list=[]
            for i in range(len(sample)):
                if sample[i][3] == n or sample[i][3] == n1:
                        s_vector=sample[i][2:4]
                        samples_list.append(s_vector)
            samples.append(samples_list)

    for s in range(len(samples)):
        for j in range(len(rules_base)):
            for k in range(len(rules_base[j])):
                if k%2!=0:
                    #print(rules_base[j][0][b][:-1],"\n", rules[i],"\n")
                    base.append(rules_base[j][k][0:2])
            if extract_rules.compare_rules(base, samples[s]) == True:
                count+=1
                activate_rules.append(rules_base[j])
            base=[]
    for a in range(len(activate_rules)):
        for b in range(len(activate_rules[a])):
            if b%2!=0:
                 activate_rule.append(activate_rules[a][b])
        final_rules.append(activate_rule)
        activate_rule=[]
    return final_rules, activate_rules, count

 


def test_classification(test_set_class1, test_set_class0, rules_base):
    tp=0
    fp=0
    tn=0
    fn=0
    mul_c1=[]
    mul_c0=[]
    mul_dummy1=[]
    mul_dummy0=[]
    total_list=[]
    scores_c1=[]
    scores_c0=[]

 

 


    for i in range(len(test_set_class1)):
        # print(test_set_class1[i], ".........\n")
        #print(">>")
        total_array=np.zeros(2)

 

        equal_rules, full_format_rules, count=activate_rules(test_set_class1[i], rules_base, 2)

 

        #for i in range(len(val_vectors_class1)):
            #print(">>>>>")
        for j in range(len(test_set_class1[i])):
            if test_set_class1[i][j][3] == 1:
                mul1 = test_set_class1[i][j][0]*test_set_class1[i][j][1]*test_set_class1[i][j][4]
                mul_c1.append(mul1)
            else:
                mul2 = test_set_class1[i][j][0]*test_set_class1[i][j][1]*test_set_class1[i][j][4]
                mul_c0.append(mul2)
        sum_dummy_c0=sum(mul_c0)
        sum_dummy_c1=sum(mul_c1)
        total_dummy=sum_dummy_c1-sum_dummy_c0

 

        if len(equal_rules) != 0:
            for e in range(len(equal_rules)):
                    for (test_class1, equal_rule) in itertools.product(test_set_class1[i], equal_rules[e]):
                        #print(test_class1[3], test_class1[2:4], equal_rule[0:2])
                        if  test_class1[3] == 1 and test_class1[2:4]==equal_rule[0:2]:
                                mul1 = test_class1[0]*test_class1[1]*equal_rule[2]
                                mul_c1.append(mul1)
                        elif test_class1[3] == 0 and test_class1[2:4] == equal_rule[0:2]:
                                mul0 =test_class1[0]*test_class1[1]*equal_rule[2]
                                mul_c0.append(mul0)
                    sum_c0=sum(mul_c0)
                    sum_c1=sum(mul_c1)
                    total_gd=sum_c1-sum_c0 # dyo total ena me to dummy kai ena opws to kanw twra otan ena deigma kanei match me kanona kai pairnw to mo 
                    # total=(total_gd+total_dummy)/2
                    total_list.append(total_gd)

 

            total_list += [0] * (2 - len(total_list))  
            total_array = np.where(np.array(total_list) == 0, total_dummy, (np.array(total_list) + total_dummy) / 2)
            total=max(total_array)
            total_list=[]
        else:
            total=total_dummy

 

        #print(total)
        scores_c1.append(total)

 

        if total > 0:
            tp=tp + 1
        else:
            fn= fn + 1
        mul_c0=[]
        mul_c1=[]
    for i1 in range(len(test_set_class0)):
        #print(">>")
        equal_rules, full_format_rules, count=activate_rules(test_set_class0[i1], rules_base,2)
        #print(equal_rules,count)
        #for i in range(len(val_vectors_class1)):

        for j1 in range(len(test_set_class0[i1])):
            if test_set_class0[i1][j1][3] == 1:
                mul1 = test_set_class0[i1][j1][0]*test_set_class0[i1][j1][1]*test_set_class0[i1][j1][4]
                mul_c1.append(mul1)
            else:
                mul2 = test_set_class0[i1][j1][0]*test_set_class0[i1][j1][1]*test_set_class0[i1][j1][4]
                mul_c0.append(mul2)
        sum_dummy_c0=sum(mul_c0)
        sum_dummy_c1=sum(mul_c1)
        total_dummy=sum_dummy_c1-sum_dummy_c0

 

        if len(equal_rules) != 0:

 

 

            for e1 in range(len(equal_rules)):
                    for (test_class0, equal_rule) in itertools.product(test_set_class0[i1], equal_rules[e1]):
                        if  test_class0[3] == 1 and test_class0[2:4]==equal_rule[0:2]:
                                mul1 = test_class0[0]*test_class0[1]*equal_rule[2]
                                mul_c1.append(mul1)
                        elif test_class0[3] == 0 and test_class0[2:4] == equal_rule[0:2]:
                                mul0= test_class0[0]*test_class0[1]*equal_rule[2]
                                mul_c0.append(mul0)


 

                    sum_c0=sum(mul_c0)
                    sum_c1=sum(mul_c1)
                    total_gd=sum_c1-sum_c0 # dyo total ena me to dummy kai ena opws to kanw twra otan ena deigma kanei match me kanona kai pairnw to mo 
                    total_list.append(total_gd)

 

            total_list += [0] * (2 - len(total_list))  
            total_array = np.where(np.array(total_list) == 0, total_dummy, (np.array(total_list) + total_dummy) / 2)
            total=max(total_array)
            total_list=[]
        else:
            total=total_dummy

 

        #print(sum_c1,sum_c0)
        scores_c0.append(total)
        #print(total)
        if total > 0:
            fp=fp + 1
        else:
            tn= tn + 1
        mul_c0=[]
        mul_c1=[]
    confusion_matrix=[[tp, fp],[fn, tn]]
    accuracy=((tp+tn)/(tp+tn+fp+fn))
    y_true = [1] * len(scores_c1) + [-1] * len(scores_c0)
    y_scores = scores_c1 + scores_c0
    auc = metrics.roc_auc_score(y_true, y_scores)
    #accuracy=round(accuracy,3)
    return accuracy, auc, confusion_matrix

 

 


def test_multiclass_classification(test_set, rules_base):
    accuracy=0
    mul_gd=[]
    max_sum=[]
    num_all_samples=0
    num_classes=len(test_set)

    for n_claass, data in enumerate(test_set):
        num_all_samples+=len(test_set[n_claass])

 

    for n_classes, data_test in enumerate(test_set):

        for i in range(len(data_test)):
            final_total=[]
            max_sum=[]
            total_gd=np.zeros((num_classes))
            # sum_gd=[]

 

            equal_rules, full_format_rules, count = activate_rules(data_test[i], rules_base, num_classes)

 

            sum_total=[]
            max_sum=[]
            for classes in range(num_classes):
                mul_c1=[]
                for j in range(len(data_test[i])):

                    # print(val_vectors_class1[i][j],"\n")
                    if data_test[i][j][3] == classes:
                        mul1 = data_test[i][j][0]*data_test[i][j][1]*data_test[i][j][4]
                        mul_c1.append(mul1)

 

                    
                sum_total.append(sum(mul_c1))
            for s in range(len(sum_total)):
                result = 0
                for k in range(len(sum_total)):
                    if s != k:
                        result += (sum_total[s] - sum_total[k])
                max_sum.append(result)
                    # max_sum1=max(max_sum)
                    # index_max=max_sum.index(max_sum1)
                    # if n_classes==index_max:
                    #     accuracy=accuracy+1
            if len(equal_rules) != 0 :
                for e in range(len(equal_rules)):

 

                    list_classes_rules = [equal_rules[e][c][1] for c in range(len(equal_rules[e]))] 
                    # print(list_classes_rules)
                    unique_classes_rules = list(set(list_classes_rules))

 


                    sum_gd=[]
                    for n in range(num_classes):
                        mul_gd=[]
                        for (test_class, equal_rule) in itertools.product(data_test[i], equal_rules[e]):
                            if test_class[3] == n and test_class[2:4] == equal_rule[0:2]:
                                mul1 = test_class[0]*test_class[1]*equal_rule[2]
                                mul_gd.append(mul1)

 

                        sum_gd.append(sum(mul_gd))
                    for n_class in range(num_classes):
                        # print(full_format_rules[j])
                        if full_format_rules[e][-1] == f'then Class{n_class}':
                            unique_classes_rules.remove(n_class)
                            # position_compared_class = [int(re.findall(r'\d+', string)[0]) for string in unique_classes_rules]
                            total_gd[n_class]+=sum_gd[n_class]-sum_gd[unique_classes_rules[0]]
                        unique_classes_rules = list(set(list_classes_rules))
                    # total_gd_list.append(total_gd)

 


            array_classes=np.ones(num_classes)
            final_total=list(max_sum)

            for rule in range(len(equal_rules)):

                act=full_format_rules[rule][-1]
                act_class = int(''.join(filter(str.isdigit, act)))

                array_classes[act_class]=array_classes[act_class] + 1
                final_total[act_class] = (total_gd[act_class] + max_sum[act_class])/(array_classes[act_class])

 

            max_index=max(final_total)
            index_max=final_total.index(max_index)

            if n_classes == index_max:
                accuracy=accuracy+1
    accuracy=accuracy/num_all_samples
    return accuracy


 