# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:56:42 2016

@author: ahmad
"""
import glob
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA




NoBall = []
Ball = []
Ball_upper = []
#Import No Ball images
for filename_NoBall in glob.glob ('/home/ahmad/Datasets/NaoCameraDataset/lower/img*.bmp'):
    img_data_NoBall = np.asarray(Image.open(filename_NoBall))
    x_NoBall = img_data_NoBall.ravel()
    NoBall.append(x_NoBall)
      
#Import ball images
for filename_Ball in glob.glob ('/home/ahmad/Datasets/NaoCameraDataset/lower/Ball/img*.bmp'):
    img_data_Ball = np.asarray(Image.open(filename_Ball))
    x_Ball = img_data_Ball.ravel()
    Ball.append(x_Ball)    
   
x = np.concatenate((NoBall,Ball), axis = 0)
x = np.asarray(x)
x = StandardScaler().fit_transform(x)
#x = PCA(964).fit_transform(x)
y_NoBall = np.zeros(622)-1
y_Ball = np.ones(342) 
y = np.concatenate((y_NoBall,y_Ball),axis= 0)
y_ball_upper = np.ones(647)

#Split data into 70% training, validation and 30% testing sets
train_test = StratifiedShuffleSplit(y, n_iter=1, train_size = 700  , test_size = 264)
for train_validate_index, test_index in train_test:
    X_train_Validate, X_test = x[train_validate_index], x[test_index]
    y_train_Validate, y_test = y[train_validate_index], y[test_index]
    
#gridsearch
    
#tuned_parameters = [{'kernel': ['linear'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
#clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5)  
#clf.fit(X_train_Validate, y_train_Validate)
#best_parameters = clf.best_params_
#print(best_parameters)
#grid_scores = clf.grid_scores_
#y_true, y_pred = y[test_index], clf.predict(x[test_index])
#score = clf.score(X_test,y_test)
#print(score)
    
cls = SVC(C=0.001, kernel = 'linear')
cls.fit(X_train_Validate, y_train_Validate)
score = cls.score(X_test, y_test)
print(score)      
    
     
   
