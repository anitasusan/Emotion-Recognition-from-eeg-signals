#libraries
import pickle
import os
import glob 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import feature_extraction as fe
import eeg_plot_code as ee
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import time
import os.path
import random

#fetching current working directory
path=  os.getcwd() 
path+="\\data_preprocessed_python"
try:
    all_files=glob.glob(path+ "/*.dat") # gets all data files
    #visualise eeg plots
    ee.visualise_egg(all_files)
    #data transformation and handling
    filename="train.csv"
    flag=True
    if(os.path.isfile(filename)==False):
        flag=False
    fe.data_processor(all_files,flag) 
    
    data=np.genfromtxt('train.csv',delimiter=',') 
    #viusalising plots()
    fe.viusaliselabels()
    #it can be seen that data is skewed towards higher ratings and hence
    #there can be a chance of data imbalance if not treated properly
    #grouping emotional ratings into low and high for each labels and encoding them
    encodedLabels=fe.encode_labels()
    
    accuracy_dt=[]
    precision_dt=[]
    f1score_dt=[]
    recall_dt=[]
    accuracy_svm=[]
    precision_svm=[]
    f1score_svm=[]
    recall_svm=[]
    accuracy_knn=[]
    precision_knn=[]
    f1score_knn=[]
    recall_knn=[]
    ypred_dt=[]
    ypred_svm=[]
    ypred_knn=[]
    #performing Ml on each label
    for i in range(len(encodedLabels)):
        
        data_train, data_test,label_train,label_test = train_test_split(data,encodedLabels[i],test_size=0.3,random_state=0)
        sc = StandardScaler()
        datascaled_train = sc.fit_transform(data_train)
        datascaled_test = sc.transform(data_test)
        #Decision Tree
        time_start=time.perf_counter() # to find computation time
        model = DecisionTreeClassifier(random_state=0)
        params = {'criterion': ['gini', 'entropy'], 'max_depth': range(50,100)}
        kfold = model_selection.KFold(n_splits=5, random_state=0,shuffle=True)
        grs = GridSearchCV(model, param_grid=params,cv=kfold)
        grs.fit(datascaled_train, label_train)
        sd=pd.DataFrame(grs.cv_results_)
        model=grs.best_estimator_
        best_par_dt=grs.best_params_
        ypred_dt.append(model.predict(datascaled_test))
        
        accuracy_dt.append(metrics.accuracy_score(label_test, ypred_dt[i]))
        precision_dt.append(metrics.precision_score(label_test, ypred_dt[i]))
        f1score_dt.append(metrics.f1_score(label_test, ypred_dt[i]))
        recall_dt.append(metrics.recall_score(label_test, ypred_dt[i]))
        print(f'Accuracy of {i}  for Decision Tree model is {metrics.accuracy_score(label_test, ypred_dt[i])}')
        print(f'Precision of {i} for Decision Tree model is {metrics.precision_score(label_test, ypred_dt[i])}')
        print(f'Classification Report{i} for Decision Tree model is{classification_report(label_test, ypred_dt[i])}')
        print(f'Computation time for {i} label for Decision Tree model is {(time.perf_counter() - time_start)}')
       
        #SVM 
        time_start=time.perf_counter() # to find computation time
        param_grid = [{'C': [0.5, 0.1, 1, 5, 10], 'kernel': ['linear'], 'class_weight':['balanced']},{'C': [0.5, 0.1, 1, 5, 10], 'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05,0.5],'kernel': ['rbf'], 'class_weight': ['balanced']}]
        kfold = model_selection.KFold(n_splits=5, random_state=0,shuffle=True)   
        model = SVC(random_state=0)
        grs = GridSearchCV(model, param_grid,cv=kfold)      
        grs.fit(datascaled_train, label_train)
        model_best = grs.best_estimator_
        best_par_svm=grs.best_params_
        ypred_svm.append(model_best.predict(datascaled_test))
        accuracy_svm.append(metrics.accuracy_score(label_test, ypred_svm[i]))
        precision_svm.append(metrics.precision_score(label_test, ypred_svm[i]))
        f1score_svm.append(metrics.f1_score(label_test, ypred_svm[i]))
        recall_svm.append(metrics.recall_score(label_test, ypred_svm[i]))
        print(f'Accuracy of {i}  for SVM model is {metrics.accuracy_score(label_test, ypred_svm[i])}')
        print(f'Precision of {i} for SVM model is {metrics.precision_score(label_test, ypred_svm[i])}')
        print(f'Classification Report{i} for SVM mmodel is{classification_report(label_test, ypred_svm[i])}')
        print(f'Computation time for {i} label for SVM  model is {(time.perf_counter() - time_start)}')
    
       #KNN
        time_start=time.perf_counter() # to find computation time   
        model_knn = KNeighborsClassifier()
        kfold = model_selection.KFold(n_splits=5, random_state=0,shuffle=True)
        params = {'n_neighbors': range(1,10)}
        grs = GridSearchCV(model_knn, param_grid=params,cv=kfold)
        grs.fit(datascaled_train, label_train)
        model_knn= grs.best_estimator_
        best_par_knn=grs.best_params_
        ypred_knn.append(model_knn.predict(datascaled_test))
        print(f'Classification Report{i} for KNN mmodel is{classification_report(label_test, ypred_knn[i] )}')
        accuracy_knn.append(metrics.accuracy_score(label_test, ypred_knn[i] ))
        precision_knn.append(metrics.precision_score(label_test, ypred_knn[i] ))
        f1score_knn.append(metrics.f1_score(label_test, ypred_knn[i] ))
        recall_knn.append(metrics.recall_score(label_test, ypred_knn[i]))
        print(f'Accuracy of {i}  for KNN model is {metrics.accuracy_score(label_test, ypred_knn[i] )}')
        print(f'Precision of {i} for KNN model is {metrics.precision_score(label_test, ypred_knn[i] )}')
        print(f'Computation time for {i} label for KNN model is {(time.perf_counter() - time_start)}')
       
       
    
    #grouping evaluation metrics for each label
    label=["Valence","Arousal","Dominance","Liking"]
    tuple_listdt=list(zip(label,accuracy_dt,precision_dt,f1score_dt,recall_dt))
    dt_metrics=pd.DataFrame(tuple_listdt,columns=["Label","Accuracy","Precision","F1Score","Recall"])
    tuple_listsvm=list(zip(label,accuracy_svm,precision_svm,f1score_svm,recall_svm))
    svm_metrics=pd.DataFrame(tuple_listsvm,columns=["Label","Accuracy","Precision","F1Score","Recall"])
    tuple_listknn=list(zip(label,accuracy_knn,precision_knn,f1score_knn,recall_knn))
    knn_metrics=pd.DataFrame(tuple_listknn,columns=["Label","Accuracy","Precision","F1Score","Recall"])
    
    #saving dataframes
    
    import dataframe_image 
    dataframe_image.export(dt_metrics, "dt_metrics.png")
    dataframe_image.export(svm_metrics, "svm_metrics.png")
    dataframe_image.export(knn_metrics, "knn_metrics.png")
    
    #identifying emotions from predicted values of each model
    em_dt=fe.identifyemotions(ypred_dt)
    k=fe.count_emotions(em_dt)
    ee.viusaliseemotions(k,"Decision Tree")
    em_svm=fe.identifyemotions(ypred_svm)
    k=fe.count_emotions(em_svm)
    ee.viusaliseemotions(k,"SVM")
    em_knn=fe.identifyemotions(ypred_knn)
    k=fe.count_emotions(em_knn)
    ee.viusaliseemotions(k,"KNN")
    tuple_em=list(zip(em_dt,em_svm,em_knn))
    em_df=pd.DataFrame(tuple_em,columns=["Decision Tree","SVM","KNN"])
except:
    print("exception occured in main")


