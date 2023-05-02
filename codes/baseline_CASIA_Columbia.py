#!/usr/bin/env python
# coding: utf-8

# Code implementation of - **A. Alahmadi, M. Hussain, H. Aboalsamh, G. Muhammad, G. Bebis, and H. Mathkour, “Passive detection of image forgery using dct and local binary pattern,” Signal, Image and Video Processing, vol. 11, no. 1, pp. 81–88, Jan 2017. [Online]. Available: https://doi.org/10.1007/s11760-016-0899-0**

# In[ ]:


import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Block-size (non-overlapping)
split_width = 16
split_height = 16

#LBP parameters
P = 8
R = 1.0


# In[ ]:


#function for dividing image into blocks
def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


# In[58]:


#main function to extract the features.
def feature_extraction(path_to_folder, class_label):
    for file_name in os.listdir(path_to_folder):
        path_to_img = os.path.join(path_to_folder,file_name)
        img = cv2.imread(path_to_img)
        if np.shape(img) == ():
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) #changing to YCrCb color space.
        img = img[:,:,1] # the Cr channel only.
        img_h, img_w = img.shape
        
        X_points = start_points(img_w, split_width, 0.0)
        Y_points = start_points(img_h, split_height, 0.0)
        
        dct_blocks=[]
        for i in Y_points:
            for j in X_points:
                block = img[i:i+split_height, j:j+split_width] # contains the non-overlapping block 
                lbp = local_binary_pattern(block, P, R, method='default')
                dct_block = dct(lbp, type=2, n=None, axis=-1, norm=None, overwrite_x=False)
                dct_blocks.append(dct_block)

        dct_blocks_array=np.asarray(dct_blocks)
        
        _,r,c=dct_blocks_array.shape
        
        img_std_list=[] #length should be r*c i.e 16*16=256 in our case.
        with_name_list=[file_name,class_label]
        
        for x in range(r):
            for y in range(c):
                pixel_depth_subarr=dct_blocks_array[:,x,y]
                std=np.std(pixel_depth_subarr)
                img_std_list.append(std)
                with_name_list.append(std)
        
        #name_list.append(file_name) 
        feature_vector.append(img_std_list)
        label.append(class_label)
        dataframe_list.append(with_name_list)


# In[ ]:





# In[59]:


feature_vector=[]
label=[]
# name_list=[]
dataframe_list=[]

# #CASIA V1.0 dataset
au_path = "E:\FinalProject\learning\dataset\working\CASIA1\Au" # set your path accordingly
tp_path1 = "E:\FinalProject\learning\dataset\working\CASIA1\Modified Tp\Tp\CM"
tp_path2 = "E:\FinalProject\learning\dataset\working\CASIA1\Modified Tp\Tp\Sp"
feature_extraction(au_path, 0)
feature_extraction(tp_path1, 1)
feature_extraction(tp_path2, 1)

# Columbia dataset
# au_path="YOUR_PATH/Columbia_ImSpliceDataset/authentic"
# tp_path="YOUR_PATH/Columbia_ImSpliceDataset/spliced"
# feature_extraction(au_path, 0)
# feature_extraction(tp_path, 1)

# CASIA V2.0 Dataset
# au_path="YOUR_PATH/CASIA2.0_revised/Au"
# tp_path="YOUR_PATH/CASIA2.0_revised/Tp"
# feature_extraction(au_path, 0)
# feature_extraction(tp_path, 1)


# In[55]:


print("Length/Dimension of features", len(feature_vector[0]))
print("Length of feature vector", len(feature_vector))
print("length of label",len(label))


# In[56]:


df=pd.DataFrame(dataframe_list)
df.rename(columns = {0: "image_names"}, inplace = True)
# df['label']=label #To add label column as well

df.head()


# In[ ]:


scaler_norm = MinMaxScaler() 
df.iloc[:,1:] = scaler_norm.fit_transform(df.iloc[:,1:].to_numpy()) # Normalising the values in dataframe.


# In[ ]:


# path_csv="CASIA2_feature.csv"
path_csv="CASIA1_feature.csv"
# path_csv="Columbia_feature.csv"

df.to_csv(path_csv) #saving dataframe to csv.


# ## Classification using the extracted features

# In[ ]:


#df=pd.read_csv('YOUR_PATH/___features.csv')

array=df.values
x_feature=array[:,2:]
y_label=array[:,1].astype('int')
print(x_feature.shape)
print(y_label.shape)


# In[54]:


# Split the data
X_train,X_test,Y_train,Y_test=train_test_split(x_feature,y_label,test_size=0.20,random_state=7)
Y_train


# In[ ]:


model_SVC=SVC(C=32,kernel='rbf',gamma=0.03125)

# Random check
kfold=KFold(n_splits=10)
#cv_results=cross_val_score(model_SVC,X_train_norm,Y_train,cv=kfold,scoring='accuracy')
#msg="%s %f (%f)" % ('Training Accuracy: ',cv_results.mean(),cv_results.std())
#print(msg)


# In[53]:


model_SVC = SVC(C=32,gamma=0.03125, kernel='rbf') #Can also try for GridSearch yourself
model_SVC.fit(X_train,Y_train) 

predictions=model_SVC.predict(X_test)

print(accuracy_score(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
accuracy_score(Y_test,predictions)


# In[ ]:


import os
os.sys.path


# In[ ]:





# In[ ]:




