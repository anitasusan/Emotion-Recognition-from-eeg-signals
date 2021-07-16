import pickle
import os
import glob 
import pandas as pd
import mne
import matplotlib.pyplot as plt

#Reading datafiles
def visualise_egg(all_files):
    try:
        i=0
        datasets=[]
        features=[]
        for files in all_files:
            with open(files, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                p = u.load()
                i=0
                for key in p.keys():
                    if i==1:
                        datasets.append(p[key])
                    else:
                        features.append(p[key])
                    i=i+1
                    
        a=datasets[0]
        b=pd.DataFrame(a[0])  
        b=b.iloc[0:32]
        sfreq=128
        ch_names=["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2","P4","P8","PO4","O2"]
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        channel_type="eeg"
        info=mne.create_info(ch_names=ch_names, ch_types=channel_type,sfreq=sfreq,montage=ten_twenty_montage)
        raw=mne.io.RawArray(b,info)
        raw.set_eeg_reference(ref_channels=ch_names)
        raw.plot_sensors(ch_type='eeg',show_names=True)
        scalings = 'auto' 
        raw.plot(n_channels=32, scalings=scalings, title='Auto-scaled Data from eg signals',show=True, block=True)
    except:
        print("Exception occured outer") 
        
def viusaliseemotions(k,model):
    try:
        max1=max(k.loc[:,'Count'])
        n=k.index[k['Count']==max1].tolist()
        explode1=[]
        for j in range(len(k)):
            flag=0
            for m in range(len(n)):
                if j==n[m]:
                    flag=1
                    break
                else:
                    flag=0
            if flag==1:
                explode1.append(0.2)
            else:
                explode1.append(0)
        fig1, ax1 = plt.subplots()
        ax1.pie(k.loc[:,'Count'],explode=explode1,labels=k.loc[:,'Emotion'], autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Emotional Wheel Slice for predicted values of {model}', loc="center")
        plt.show()
    except:
        print("Exception occured outer")