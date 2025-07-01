
import os, random
from pathlib import Path
from typing import List
import numpy as np
import scipy.signal as spsig
from scipy import interpolate
from scipy.stats import skew, kurtosis
import pywt, wfdb, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

SEED = 0
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
SEGMENT_SECONDS = 10.0
OVERLAP = 0.5
NORMAL_DIR = "./Norm data"
PAF_DIR = "./Paf data"

# ---------------- preprocessing ----------------

def preprocess_ecg(sig: np.ndarray, fs: int) -> np.ndarray:
    b, a = spsig.butter(3, 0.5/(fs/2), 'high'); sig = spsig.filtfilt(b,a,sig)
    for f in (50,60):
        if f<fs/2:
            b,a=spsig.iirnotch(f/(fs/2),30); sig=spsig.filtfilt(b,a,sig)
    b, a = spsig.butter(4, 40/(fs/2), 'low'); sig = spsig.filtfilt(b,a,sig)
    coeffs=pywt.wavedec(sig,'sym8',level=4)
    for i in range(1,len(coeffs)):
        thr=np.sqrt(2*np.log(len(sig)))*np.median(np.abs(coeffs[i]))/0.6745
        coeffs[i]=pywt.threshold(coeffs[i],thr,'soft')
    sig=pywt.waverec(coeffs,'sym8')[:len(sig)]
    return (sig-sig.mean())/(sig.std()+1e-6)

# ---------------- window slicing --------------

def slice_windows(sig: np.ndarray, fs:int)->np.ndarray:
    win=int(SEGMENT_SECONDS*fs); step=max(int(win*(1-OVERLAP)),1)
    if len(sig)<win: return np.empty((0,win),np.float32)
    return np.asarray([sig[s:s+win] for s in range(0,len(sig)-win+1,step)],np.float32)

# ---------------- feature extraction ----------

def detect_qrs_peaks(sig,fs):
    ny=fs/2;b,a=spsig.butter(3,[5/ny,15/ny],'band');f=spsig.lfilter(b,a,sig)
    der=np.append(np.diff(f),0); sq=der**2; mw=np.convolve(sq,np.ones(int(0.15*fs))/int(0.15*fs),'same')
    p,_=spsig.find_peaks(mw,distance=int(0.2*fs))
    if len(p): p,_=spsig.find_peaks(mw,height=0.5*np.mean(mw[p]),distance=int(0.2*fs))
    return p

def extract_hrv(sig,fs):
    p=detect_qrs_peaks(sig,fs)
    if len(p)<3: return None
    rr=np.diff(p)/fs; rr=rr[(rr>=0.4)&(rr<=2.0)]
    if len(rr)<3: return None
    hr=60/rr.mean(); sdnn=rr.std(); rmssd=np.sqrt(np.mean(np.diff(rr)**2)); pnn50=100*np.sum(np.abs(np.diff(rr))>0.05)/len(rr)
    return dict(mean_hr=hr,sdnn=sdnn,rmssd=rmssd,pnn50=pnn50)

def extract_features(win,fs):
    mean,std,p2p=win.mean(),win.std(),win.max()-win.min(); sk,k=skew(win),kurtosis(win)
    f,psd=spsig.welch(win,fs,nperseg=min(256,len(win))); tot=psd.sum(); tot=tot if tot>0 else 1
    p0_5=psd[f<=5].sum()/tot; p5_15=psd[(f>5)&(f<=15)].sum()/tot; p15_50=psd[f>15].sum()/tot
    ent=-np.sum((psd/tot)*np.log2(psd/tot+1e-12))
    hrv=extract_hrv(win,fs) or {}
    feat=[mean,std,sk,k,p2p,p0_5,p5_15,p15_50,ent,
          hrv.get('mean_hr',0),hrv.get('sdnn',0),hrv.get('rmssd',0),hrv.get('pnn50',0),0,0,0]
    return np.asarray(feat,np.float32)

# ---------------- data loaders ---------------

def load_record(h:Path):
    rec=wfdb.rdrecord(str(h.with_suffix(''))); sig=rec.p_signal.astype(np.float32)
    if sig.ndim==2 and sig.shape[1]>1: sig=sig[:,0]
    return preprocess_ecg(sig,int(rec.fs)), int(rec.fs)

def build_dataset(headers:List[Path], ref_fs:int):
    wins,feats,labs=[],[],[]
    for h in headers:
        sig,fs=load_record(h); lab=int(h.name[0].lower()=='p')
        if fs!=ref_fs: continue
        for w in slice_windows(sig,fs):
            wins.append(w); feats.append(extract_features(w,fs)); labs.append(lab)
    Xw=np.asarray(wins,np.float32)[...,None]; Xf=np.asarray(feats,np.float32); y=np.asarray(labs,int)
    return Xw,Xf,y

# ---------------- model ----------------------

def build_hybrid(win_len,n_feat):
    sig_in=Input(shape=(win_len,1)); feat_in=Input(shape=(n_feat,))
    x=Conv1D(64,7,activation='relu')(sig_in); x=MaxPooling1D(3)(x); x=Dropout(0.2)(x)
    x=Conv1D(128,7,activation='relu')(x); x=MaxPooling1D(3)(x); x=Dropout(0.2)(x)
    x=Conv1D(256,7,activation='relu')(x); x=MaxPooling1D(3)(x); x=Dropout(0.2)(x)
    x=Flatten()(x); x=Dense(128,activation='relu')(x); x=Dropout(0.5)(x)
    y=Dense(64,activation='relu')(feat_in); y=Dropout(0.3)(y); y=Dense(32,activation='relu')(y)
    z=Concatenate()([x,y]); z=Dense(64,activation='relu')(z); z=Dropout(0.3)(z)
    out=Dense(1,activation='sigmoid')(z)
    model=Model([sig_in,feat_in],out)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return model

# ---------------- main -----------------------

def main():
    norm=list(Path(NORMAL_DIR).glob('*.hea')); paf=list(Path(PAF_DIR).glob('*.hea'))
    if not norm or not paf: raise SystemExit('No data found')
    ref_fs=load_record(norm[0])[1]; win_len=int(ref_fs*SEGMENT_SECONDS)
    headers=np.array(norm+paf); labels=np.array([0]*len(norm)+[1]*len(paf))
    h_tr,h_te,_,_=train_test_split(headers,labels,test_size=0.2,random_state=SEED,stratify=labels)
    Xtr_w,Xtr_f,ytr=build_dataset(h_tr,ref_fs); Xte_w,Xte_f,yte=build_dataset(h_te,ref_fs)
    model=build_hybrid(win_len,Xtr_f.shape[1]); model.summary()
    cbs=[EarlyStopping(patience=10,restore_best_weights=True,monitor='val_loss'),
         ReduceLROnPlateau(factor=0.5,patience=4,monitor='val_loss'),
         ModelCheckpoint('best_model.keras',save_best_only=True,monitor='val_loss')]
    hist=model.fit([Xtr_w,Xtr_f],ytr,validation_split=0.2,epochs=60,batch_size=64,shuffle=True,callbacks=cbs,verbose=1)
    loss,acc,auc=model.evaluate([Xte_w,Xte_f],yte,verbose=0)
    print(f'Test acc={acc:.3f} auc={auc:.3f}')
    y_pred=(model.predict([Xte_w,Xte_f],verbose=0).squeeze()>0.5).astype(int)
    print('\n',classification_report(yte,y_pred,target_names=['Normal','PAF']))
    cm=confusion_matrix(yte,y_pred)
    plt.imshow(cm,cmap='Blues'); plt.title('Confusion'); plt.colorbar(); plt.savefig('confusion.png'); plt.close()
    model.save('ecg_paf_detector.keras')

if __name__=='__main__':
    main()

