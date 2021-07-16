
import csv
from collections import defaultdict
import numpy as np
import pywt
import os
import os.path
#Reading datafiles

def fttransformation(): 
    try:  
        fout_data = open("train.csv",'a')
        #vec = []
        chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
        columns = defaultdict(list) # each value in each column is appended to a list    
        with open("features_raw.csv") as f:
            reader = csv.DictReader(f) # read rows into a dictionary format
            try:
                for row in reader: 
                    for (k,v) in row.items():
                       if v is not None:
                           columns[k].append(v);
            except:
                print("Exception Occured inner loop")
                
                
        for i in chan:
        	x = np.array(columns[i]).astype(np.float)
        	coeffs = pywt.wavedec(x, 'db4', level=6)
        	cA6, cD6, cD5,cD4,cD3,cD2,cD1 = coeffs
        	cD5 = np.std(cD5)
        	cD4 = np.std(cD4)
        	cD3 = np.std(cD3)
        	cD2 = np.std(cD2)
        	cD1 = np.std(cD1)
        	if i =="O2":
        		fout_data.write(str(cD5)+",")
        		fout_data.write(str(cD4)+",")
        		fout_data.write(str(cD3)+",")
        		fout_data.write(str(cD2)+",")
        		fout_data.write(str(cD1))
        	else:
        		fout_data.write(str(cD5)+",")
        		fout_data.write(str(cD4)+",")
        		fout_data.write(str(cD3)+",")
        		fout_data.write(str(cD2)+",")
        		fout_data.write(str(cD1)+",")
        fout_data.write("\n")
        fout_data.close()
    except:
        print("Exception occured fft")
    finally:
        fout_data.close()