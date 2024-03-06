import my_preprocessing
import clustering
import similarities
import my_fuzzification
import features
import my_classification
import extract_rules
import gradient_descent
import importlib
import time

importlib.reload(features)
importlib.reload(extract_rules)
importlib.reload(my_preprocessing)
importlib.reload(clustering)
importlib.reload(similarities)
importlib.reload(my_classification)
importlib.reload(my_fuzzification)
importlib.reload(gradient_descent)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys







def call_all_functions(df, class_name, num_iters, learning_rate, split_val_set, n_clusters,n_fuzzy_sets):
    accuracy=[]
    auc=[]
    auc_gd=[]
    accuracy_gd=[]
    num_folds=10
    array_classes=np.zeros(len(np.unique(class_name)))
    n_rules_per_class1=[]
    time_gd=[]
    time__cluster_sim=[]
    time_features=[]
    time_dummy=[]

    # scaled_df=my_preprocessing.standardScaler_array_transform(df)
    # print(scaled_df)


    scaled_df=df.astype(np.float64)



    classes=clustering.separate_classes2(scaled_df,class_name)
    # sys.exit()
    classes_indexes = my_preprocessing.kfold_cross_validation(num_folds, classes)
    # print(classes_indexes)
    for f in range(num_folds):

        print(">>>>>>>", f+1 ,"fold of cross validation >>>>>>>")

        # classes_indexes=my_preprocessing.kfold_cross_validation(num_folds,classes)
        # print(classes_indexes)

        test_set = np.empty(len(classes_indexes), dtype=object)
        train_set = np.empty(len(classes_indexes), dtype=object)
        # print(class1[0])


        for i, (indexes,data_classes) in enumerate(zip(classes_indexes,classes)):
            test_index=indexes[f][1]
            train_index = indexes[f][0]
            test_set[i]=data_classes[test_index]
            # train_set[i]=np.delete(data_classes, test_index, axis=0)
            train_set[i] =data_classes[train_index]

    
        train_sets, val_sets=my_preprocessing.train_test_split1(train_set, split_val_set)
        mean,std, train_sets=my_preprocessing.my_zscore_train(train_sets)
        test_set=my_preprocessing.my_zscore_valtest(test_set, mean, std)
        val_sets=my_preprocessing.my_zscore_valtest(val_sets, mean, std)

        sample_array = []
        for i in range(len(val_sets)):
            sample_array.append(np.zeros((len(val_sets[i]), 1)) + i)


        sample_val = np.vstack(sample_array)




        

          
        sample_test_array = []
        for i in range(len(test_set)):
            sample_test_array.append(np.zeros((len(test_set[i]), 1)) + i)

        
        sample_test=np.vstack(sample_test_array)


        
       


        #First clustering with kmeans and take the medoids

        start_time=time.time()
        
        class_centers, centers=clustering.my_clustering(train_sets, n_clusters)
        


        ###################################################################

        #Calculation of similarities with Euclidean Distance
        train_similarities=similarities.similarity_measure_vector1(train_sets,centers)
        
    
        test_similarities=similarities.similarity_measure_vector1(test_set,centers)
        val_similarities=similarities.similarity_measure_vector1(val_sets,centers)
        end_time = time.time()
        time__=end_time-start_time
        time__cluster_sim.append(time__)




        #Min-Max normalization of the similarities
        class_train_normalized,min_val,max_val=my_preprocessing.minmaxnormalization_1(train_similarities)
        val_class_normalized=my_preprocessing.minmaxnormalization_2(val_similarities,min_val,max_val)
        test_class_normalized=my_preprocessing.minmaxnormalization_2(test_similarities,min_val,max_val)



        #Connect classes with the centers of each class
        
        class_similarities_class_centers, n_cluster_thesis=similarities.connect_class_with_class_centers(class_train_normalized,n_clusters)
        # print(class_similarities_class_centers)


        #Create the fuzzy sets with the desired number

        fuzzy_sets_class=my_fuzzification.fuzzification(class_similarities_class_centers, n_fuzzy_sets)


        

        #Lists of similaririties and membership values
        list_vectors_similarities1=my_fuzzification.vectorization_similarities(val_class_normalized, fuzzy_sets_class)
        # print(list_vectors_similarities1[1],"++\n" )



        list_vectors_similarities1_test=my_fuzzification.vectorization_similarities(test_class_normalized, fuzzy_sets_class)
        # print(list_vectors_similarities1)
        # sys.exit()
 
        #Keep the necessary vectors
        st=time.time()
        
        vectors_weights_val_class=features.feature_selection(list_vectors_similarities1,n_cluster_thesis,n_fuzzy_sets)
      
        
        vectors_weights_test_class=features.feature_selection(list_vectors_similarities1_test,n_cluster_thesis,n_fuzzy_sets)
        end=time.time()

        time_f=end-st
        time_features.append(time_f)



        
  
        
        #Accuracy of dummy classification
        print(">>>>>>>> Dummy classification of ",f+1, "fold >>>>>>>>>>>")
        
        s=time.time()
        if len(vectors_weights_test_class)==2:
            accuracy1, auc1, cm=my_classification.dummy_classification(vectors_weights_test_class[1], vectors_weights_test_class[0])
            auc.append(auc1)

        else:
            accuracy1=my_classification.dummy_multiclass_classification(vectors_weights_test_class)
        e=time.time()
        timee=e-s
        time_dummy.append(timee)


        accuracy.append(accuracy1)


        #Create the rules base
        # print(vectors_weights_val_class,"_____________\n")
        rules_basee=extract_rules.keep_unique_rules1(vectors_weights_val_class)
        
        for n_class_rule_base, rules_per_class in enumerate(rules_basee):
            array_classes[n_class_rule_base] = len(rules_per_class)

        n_class_rule_base=np.mean(array_classes)

        n_rules_per_class1.append(n_class_rule_base)


        
        #Iterations and update weights according to gradient descent method


        print(">>>>>>>>>Gradient descent method's iterations at the",f+1, "fold >>>>>>>>>>>>>>>>")
        
        start=time.time()
        rules_base, error =gradient_descent.update_weights_gd(rules_basee, vectors_weights_val_class, learning_rate, sample_val, num_iters, n_fuzzy_sets)
        end2=time.time()
        time_g=end2-start
        time_gd.append(time_g)


        time_acc_gd=[]
        start1=time.time()
        if len(vectors_weights_test_class)==2:
            accuracy_gd1, auc2, cm2=gradient_descent.test_classification(vectors_weights_test_class[1], vectors_weights_test_class[0], rules_base)
            auc_gd.append(auc2)

        else:
            accuracy_gd1=gradient_descent.test_multiclass_classification(vectors_weights_test_class, rules_base)

        end3=time.time()
        time_a=end3-start1
        time_acc_gd.append(time_a)
        accuracy_gd.append(accuracy_gd1)



    mean_time_clust_sim=np.mean(time__cluster_sim)
    mean_time_feature_sel=np.mean(time_features)
    mean_time_dummy=np.mean(time_dummy)
    mean_time_gd=np.mean(time_gd)
    mean_time_acc_gd=np.mean(time_acc_gd)
    accuracy_mean=np.mean(accuracy)
    auc_mean=np.mean(auc)
    auc_std=np.std(auc)
    accuracy_gd_mean=np.mean(accuracy_gd)
    accuracy_std=np.std(accuracy)
    accuracy_gd_std=np.std(accuracy_gd)
    auc_gd_mean=np.mean(auc_gd)
    auc_gd_std=np.std(auc_gd)
    rules_mean=np.mean(n_rules_per_class1)
    print(rules_mean)
    print(f"Mean time of 10 folds taken for first clustering and similarities calculation: {mean_time_clust_sim:.3f} seconds")
    print(f"Mean time of 10 folds taken for feature selection: {mean_time_feature_sel:.3f} seconds")
    print(f"Mean time of 10 folds taken for dummy classification: {mean_time_dummy:.3f} seconds")
    print(f"Mean time of 10 folds taken for training with GD: {mean_time_gd:.3f} seconds")
    print(f"Mean time of 10 folds taken for GD classification with RB: {mean_time_acc_gd:.3f} seconds")

    return accuracy, accuracy_mean, accuracy_std, accuracy_gd_mean, accuracy_gd_std, auc_mean, auc_std, auc_gd_mean, auc_gd_std, rules_mean
















