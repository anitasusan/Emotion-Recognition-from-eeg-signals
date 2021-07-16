import pickle
import fft
import numpy as np
from sklearn import preprocessing
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import os.path

def data_processor(all_files,flag):
    try:
        chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
        nTrial,nChannel, nTime  = 40,32, 8064
        fout_labels0 = open("labels_0.csv",'w')
        fout_labels1 = open("labels_1.csv",'w')
        fout_labels2 = open("labels_2.csv",'w')
        fout_labels3 = open("labels_3.csv",'w')
        for files in all_files:
            with open(files,'rb') as f:
                p=pickle.load(f,encoding="Latin1")
                #p = u.load()
            for tr in range(nTrial):
                fout_data = open("features_raw.csv",'w')
                for ch in chan:
                    fout_data.write(ch+",")
                fout_data.write("\n")
                for dat in range(nTime):
                    for ch in range(nChannel):
                        if ch==31:
                            fout_data.write(str(p['data'][tr][ch][dat]))
                        else:
                            fout_data.write(str(p['data'][tr][ch][dat])+",")
                    fout_data.write("\n")
                fout_data.close()
                if(flag==False):
                    fft.fttransformation()
                fout_labels0.write(str(p['labels'][tr][0]) + "\n")
                fout_labels1.write(str(p['labels'][tr][1]) + "\n")
                fout_labels2.write(str(p['labels'][tr][2]) + "\n")
                fout_labels3.write(str(p['labels'][tr][3]) + "\n");
            print(files)
        fout_labels0.close()
        fout_labels1.close()
        fout_labels2.close()
        fout_labels3.close()
    except:
        print("Exception occured data_processor")
        
def group_labels():
    try:
        
        #grouping emotional ratings into low and high for each labels
        label0 = np.genfromtxt('labels_0.csv',delimiter=',')
        label1 = np.genfromtxt('labels_1.csv',delimiter=',')
        label2 = np.genfromtxt('labels_2.csv',delimiter=',')
        label3 = np.genfromtxt('labels_3.csv',delimiter=',')
        labels_array=[label0,label1,label2,label3]
        if labels_array== None:
            coded_features=[]
        else:
            coded_features=[[] for _ in range(len(labels_array))]
            for item in range(len(labels_array)):
                for k in range(len(labels_array[item])):
                  if(labels_array[item][k]>=0 and labels_array[item][k]<5):
                      coded_features[item].append('low')
                  # elif(labels_array[item][k]>=3 and labels_array[item][k]<6):
                  #     coded_features[item].append('neutral')
                  else:
                      coded_features[item].append('high') 
    except:
        print("Exception occured group_labels")
    return coded_features

def decode_labels(i,y_pred,label_test):
    i=0
    try:
        filename="pred_labels"+str(i)+".csv"
        fout_label=open(filename,'w')
        fout_label.write("Predicted,Actual\n")
        for k in range(len(y_pred)):
            if y_pred[k]==1:
                fout_label.write("High"+","+str(label_test[k])+"\n")
            else:
                fout_label.write("Low"+","+str(label_test[k])+"\n")
        fout_label.close()
        
     except:
        print("Exception occured decode_labels")   
        
def encode_labels():
    try:
        
        coded_features=group_labels()
        #Encoding labels
        LabelEncoder = preprocessing.LabelEncoder()
        if LabelEncoder== None:
            encodedLabels=[]
        else:
            encodedLabels=[[] for _ in range(len(coded_features))]
            for k in range(len(coded_features)):
                encoded=LabelEncoder.fit_transform(coded_features[k])
                encodedLabels[k]=encoded.astype(int)
    except:
        print("Exception occured outer")
    return encodedLabels


def viusaliselabels():
    try:
            
        label0 = np.genfromtxt('labels_0.csv',delimiter=',')
        label1 = np.genfromtxt('labels_1.csv',delimiter=',')
        label2 = np.genfromtxt('labels_2.csv',delimiter=',')
        label3 = np.genfromtxt('labels_3.csv',delimiter=',')
        labels_array=[label0,label1,label2,label3]
        for item in range(len(labels_array)):
            plt.hist(labels_array[item])
            plt.xlabel(f'Label number {item}')
            plt.ylabel("Count")
            plt.title(f'Variation of ratings for label {item}')
            plt.show()
     except:
        print("Exception occured label visualisation")
        
def identifyemotions(ypred_knn):

    emotion=[]
    try:
        val=[]
        for each in ypred_knn[0]:
            if each==0:
                val.append("sad")
            elif each==1:
                val.append("happy")
        emotion.append(val)
        val=[]
        for each in ypred_knn[1]:
            if each==0:
                val.append("calm")
            elif each==1:
                val.append("aroused")
        emotion.append(val)
        val=[]
        for each in ypred_knn[2]:
            if each==0:
                val.append("compliant")
            elif each==1:
                val.append("dominant")
        emotion.append(val)
        val=[]
        for each in ypred_knn[2]:
            if each==0:
                val.append("disliked")
            elif each==1:
                val.append("liked")
        emotion.append(val) 
    except:
        print("Exception occured identify emotions)
    return emotion   

def count_emotions(em_dt):
    emotion=[]
    count=[]
    try:
        for each in range(len(em_dt)):
            values, counts = np.unique(em_dt[each], return_counts=True)
            for k in range(len(values)):
                emotion.append(values[k])
                count.append(counts[k])
        list_em=list(zip(emotion,count))
        em_dt=pd.DataFrame(list_em,columns=["Emotion","Count"])
    except:
        print("Exception occured in count_emotions")
    if em_dt==None:
        return None
    else:
        return em_dt