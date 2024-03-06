from operator import concat
import pandas as pd
import numpy as np
import re

from collections import Counter
from itertools import chain
import itertools
# def compare(s, t):
#     return Counter(list(chain(*s))) == Counter(list(chain(*t)))

def compare_rules(list1, list2):
    if len(list1) != len(list2):
        return False

    copy_list1 = list(list1)
    copy_list2 = list(list2)

    # Sort the list elements individually
    sorted_list1 = sorted(copy_list1, key=lambda x: str(x))
    sorted_list2 = sorted(copy_list2, key=lambda x: str(x))

    for i in range(len(sorted_list1)):
        if isinstance(sorted_list1[i], list) and isinstance(sorted_list2[i], list):
            if not compare_rules(sorted_list1[i], sorted_list2[i]):
                return False
        elif sorted_list1[i] != sorted_list2[i]:
            return False

    return True

def compare(list1, list2):
    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        if isinstance(list1[i], list) and isinstance(list2[i], list):
            if not compare_rules(list1[i], list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False

    return True


def keep_unique_rules1(vectors_val_class):
    rules_base = []
    indexes = (2, 3, 4)
    rules_list1=[]
    num_classes=len(vectors_val_class) #3
    # class_labels = np.arange(num_classes) #[0,1,2]

    for n_classes,data_classes in enumerate(vectors_val_class):
        class_labels = np.delete(np.arange(num_classes), n_classes)
        # vectors_weights_val = vectors_val_class[c]
        rules_list = ['if']
        rules = []
        for i in range(len(data_classes)):
            rules_list1=[]
            for a in range(num_classes-1):
                rules_list=['if']
                flag=class_labels[a]
                for j in range(len(data_classes[i])):
                    if (data_classes[i][j][3] == n_classes or data_classes[i][j][3] == flag) :
                        selected_indexes = [data_classes[i][j][k] for k in indexes]
                        rules_list.append(selected_indexes)
                        rules_list.append('and')
                rules_list.pop()
                # rules_list1.append(rules_list)
                rules_list.append(f'then Class{n_classes}')

                # rules_lst = []
                rules_list1.append(rules_list)
            # rules_list1.append(rules_list)
            rules.extend(rules_list1)
            # print(rules)

        unique_rules = []
        for r in range(len(rules)):
            is_duplicate = False
            for r2 in range(r + 1, len(rules)):
                

                if compare_rules(rules[r], rules[r2]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_rules.append(rules[r])

        rules_base_c = [sublist for sublist in unique_rules]
        rules_base.append(rules_base_c)


    
    return tuple(rules_base)




def create_arrays_rules(rules_base):
    class_per_rule=[]
    class_compare1_final=[]
    for n_class, rules in enumerate(rules_base):
        for r in range(len(rules)):
            class_compare1=[]
            class_compare=[]
            clas=rules[r][-1]
            result = re.search(r'\d+', clas)
            extracted_integer = int(result.group())
            class_per_rule.append(extracted_integer)
            for r1 in range(len(rules[r])):
                if r1%2!=0:
                    class_compare.append(rules[r][r1][1])
            class_compare1=list(set(class_compare))
            class_compare1_final.append(class_compare1)

        
        
    class_per_rule = np.array(class_per_rule).reshape(len(class_per_rule), 1)
    classes_compare = np.array(class_compare1_final)
        


        
    return class_per_rule,classes_compare



def create_array_signs(rules_base, class_per_rule, class_compare, num_fuzzy_sets):
    size_rules=class_per_rule.shape[0]
    size_fuzzy=num_fuzzy_sets
    size=(size_rules,size_fuzzy)
    size2=(size_rules,2*size_fuzzy)
    array_signs_1 = np.zeros(size)    
    array_signs_2 = np.zeros(size)

    array_weights_1 = np.zeros(size)
    array_weights_2 = np.zeros(size)
    array_signs=np.zeros(size2)
    array_weights=np.zeros(size2)
    all_rules=[]
    for i,rules_per_class in enumerate(rules_base):
        all_rules+=rules_per_class




    for r in range(len(all_rules)):
        for r1 in range(len(all_rules[r])):
                
            if r1%2!=0 and all_rules[r][r1][1] == class_per_rule[r][0]:
                pos=int(all_rules[r][r1][0])
                array_signs_1[r,pos]=1
                array_weights_1[r,pos]=all_rules[r][r1][2]
            elif r1%2!=0 and all_rules[r][r1][1] != class_per_rule[r][0]:
                pos1=int(all_rules[r][r1][0])
                array_signs_2[r,pos1]=-1
                array_weights_2[r,pos1]=all_rules[r][r1][2]

        

        if class_per_rule[r][0] == class_compare[r][0]:
            array_signs[r] = np.concatenate((array_signs_1[r], array_signs_2[r]), axis=0)
            array_weights[r] = np.concatenate((array_weights_1[r], array_weights_2[r]), axis=0)
        else:
            array_signs[r] = np.concatenate((array_signs_2[r], array_signs_1[r]), axis=0)
            array_weights[r] = np.concatenate((array_weights_2[r], array_weights_1[r]), axis=0)

    

            

    return array_signs, array_weights


            








