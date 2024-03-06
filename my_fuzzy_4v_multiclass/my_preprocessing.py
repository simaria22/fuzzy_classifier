import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def create_array_classes(sets):
    sample_array = []
    for i in range(len(sets)):
        sample_array.append(np.zeros((len(sets[i]), 1)) + i)
    return sample_array
    
   
    
   
def standardScaler_array_transform(array):
    ss = StandardScaler()
    scaled_array = ss.fit_transform(array)

    return scaled_array

def my_zscore_train(train):

    train_concat=np.concatenate(train)

    mean=np.mean(train_concat,axis=0)
    std = np.std(train_concat,axis=0)
    std[std==0]=1

    train2=np.empty_like(train)
    for i,data in enumerate(train):

        train2[i]= (data - mean )/ std


    return mean,std, train2


def my_zscore_valtest(valtest,mean,std):

    valtest2 = np.empty_like(valtest)
    for i, data in enumerate(valtest):
        valtest2[i] = (data - mean) / std

    return valtest2



def train_test_split1(train_set, val_split):
  val_set = np.empty(len(train_set), dtype=object)
  final_train_set = np.empty(len(train_set), dtype=object)
  for i, train_sets in enumerate(train_set):
        final_train_set[i], val_set[i] = train_test_split(train_sets, test_size=val_split)
  return final_train_set,val_set
     
      
   
   

def kfold_cross_validation(n_splits, class_splits):
    kf = KFold(n_splits=n_splits,shuffle=True)
    train_test_indices = []
    classes_indexes=()

    # Loop over the folds and store the training and validation indices
    for i, df in enumerate(class_splits):
      # class_split=df[i]
      for train_index, test_index in kf.split(df):
          train_test_indices.append((train_index, test_index))
      classes_indexes+=(train_test_indices,)
      train_test_indices=[]

        #print(test_index,train_index)


        #test_set=class_splits.iloc[test_index]
        #train_set=class_splits.drop(test_index)
    return classes_indexes



def minmaxnormalization_2(classes, min_val, max_val):
    norm_classes = []
    for i in range(len(classes)):
        norm_class = (classes[i] - min_val) / (max_val - min_val)
        norm_class = np.clip(norm_class, 0, 1)
        # norm_class = norm_class.reshape(-1, norm_class.shape[-1])
        norm_classes.append(norm_class)
    return tuple(norm_classes)



#

# def minmaxnormalization_2(classes, min_val, max_val):
#     norm_classes =np.empty(len(classes),dtype=object)
#     for classes, data in enumerate(classes):
#         norm_classes[classes]=(data - min_val) / (max_val - min_val)
#         norm_classes[classes] = np.clip(norm_classes[classes], 0, 1)

#     return norm_classes



def minmaxnormalization_1(classes):
    list_scaled = []

    merged_classes = np.concatenate(classes)
    max_val = np.max(merged_classes)
    min_val = np.min(merged_classes)

    for arr_class in classes:
        normalized_class = (arr_class - min_val) / (max_val - min_val)
        normalized_class = np.clip(normalized_class, 0, 1)
        list_scaled.append(normalized_class)

    return list_scaled, min_val, max_val



# def minmaxnormalization_1(classes):
#     list_scaled = []

#     merged_classes = np.concatenate(classes)
#     max_val = np.max(merged_classes)
#     min_val = np.min(merged_classes)

#     for arr_class in classes:
#         normalized_class = (arr_class - min_val) / (max_val - min_val)
#         normalized_class = np.clip(normalized_class, 0, 1)
#         list_scaled.append(normalized_class)

#     return list_scaled, min_val, max_val

