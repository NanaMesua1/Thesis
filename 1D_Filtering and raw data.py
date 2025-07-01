
import os, random
from pathlib import Path

from typing import List, Tuple

import numpy as np, scipy.signal as spsig, pywt, wfdb, matplotlib
from scipy import interpolate
from scipy.stats import skew, kurtosis   # ‑‑ still used by filter viz helpers
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

SEED = 0
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --------------------------- parameters ----------------------------------
SEGMENT_SECONDS = 10.0
OVERLAP         = 0.5
NORMAL_DIR      = "./Norm data"
PAF_DIR         = "./Paf data"


# Full medical‑grade pre‑processing chain


def preprocess_ecg(sig: np.ndarray, fs: int,
                   hp=True, notch=True, lp=True, wavelet=True) -> np.ndarray:
    clean = sig.astype(np.float32)
    if hp:
        b,a = spsig.butter(3, 0.5/(fs/2), 'high'); clean = spsig.filtfilt(b,a,clean)
    if notch:
        for f in (50,60):
            if f < fs/2:
                b,a = spsig.iirnotch(f/(fs/2), 30); clean = spsig.filtfilt(b,a,clean)
    if lp:
        b,a = spsig.butter(4, 40/(fs/2), 'low'); clean = spsig.filtfilt(b,a,clean)
    if wavelet:
        coeffs = pywt.wavedec(clean, 'sym8', level=4)
        for i in range(1,len(coeffs)):
            thr = np.sqrt(2*np.log(len(clean))) * np.median(np.abs(coeffs[i]))/0.6745
            coeffs[i] = pywt.threshold(coeffs[i], thr, 'soft')
        clean = pywt.waverec(coeffs, 'sym8')[:len(clean)]
    return (clean-clean.mean())/(clean.std()+1e-6)


# Helper functions


def discover_records(folder: str) -> List[Path]:
    """Return sorted list of *.hea headers in a folder."""
    return sorted(Path(folder).glob("*.hea"))


def load_record(header: Path) -> Tuple[np.ndarray,int]:
    rec = wfdb.rdrecord(str(header.with_suffix("")))
    sig = rec.p_signal.astype(np.float32)
    if sig.ndim==2 and sig.shape[1]>1:
        sig = sig[:,0]
    sig = preprocess_ecg(sig, int(rec.fs))
    return sig, int(rec.fs)


def slice_windows(sig: np.ndarray, fs: int, seconds: float, overlap: float):
    win = int(seconds*fs); step = max(int(win*(1-overlap)),1)
    if len(sig)<win:
        return np.empty((0,win),np.float32)
    return np.asarray([sig[s:s+win] for s in range(0,len(sig)-win+1,step)],
                      np.float32)


# CNN model (single input)


def build_model(input_len:int) -> Sequential:
    model = Sequential([
        Conv1D(64,7,activation='relu',input_shape=(input_len,1)),
        MaxPooling1D(3), Dropout(0.2),
        Conv1D(128,7,activation='relu'), MaxPooling1D(3), Dropout(0.2),
        Conv1D(256,7,activation='relu'), MaxPooling1D(3), Dropout(0.2),
        Flatten(),
        Dense(128,activation='relu'), Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# Plot helpers


def plot_history(hist):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(hist.history['accuracy'],label='train')
    plt.plot(hist.history['val_accuracy'],label='val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1,2,2); plt.plot(hist.history['loss'],label='train')
    plt.plot(hist.history['val_loss'],label='val'); plt.title('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig('training_history.png'); plt.close()


def plot_confusion(cm):
    plt.figure(figsize=(6,5))
    plt.imshow(cm,cmap='Blues'); plt.title('Confusion matrix'); plt.colorbar()
    plt.xticks([0,1],['Normal','PAF']); plt.yticks([0,1],['Normal','PAF'])
    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            plt.text(j,i,int(cm[i,j]),ha='center',va='center',
                     color='white' if cm[i,j]>thresh else 'black')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig('confusion_matrix.png'); plt.savefig('confusion_matrix.pdf'); plt.close()


# Main pipeline


def main():
    normal = discover_records(NORMAL_DIR)
    paf    = discover_records(PAF_DIR)
    if not normal or not paf:
        raise SystemExit('No WFDB headers found. Check data folders.')
    print(f'Found {len(normal)} normal  |  {len(paf)} PAF records')

    ref_sig, ref_fs = load_record(normal[0])
    win_len = int(ref_fs*SEGMENT_SECONDS)

    headers = np.array(normal+paf)
    labels  = np.array([0]*len(normal)+[1]*len(paf))

    h_train,h_test,_,_ = train_test_split(headers,labels,test_size=0.2,
                                          random_state=SEED,stratify=labels)

    def collect(hdrs):
        xs, ys = [], []
        for h in hdrs:
            sig, fs = load_record(h)
            if fs!=ref_fs: continue
            w = slice_windows(sig,fs,SEGMENT_SECONDS,OVERLAP)
            xs.append(w); ys.extend([int(h.name[0].lower()=='p')]*len(w))
        if not xs:
            return np.empty((0,win_len,1),np.float32), np.empty((0,),int)
        X = np.concatenate(xs)[...,None]
        y = np.asarray(ys,int)
        return X,y

    Xtr, ytr = collect(h_train)
    Xte, yte = collect(h_test)
    print('Train windows:', Xtr.shape, ' Test windows:', Xte.shape)

    model = build_model(win_len)
    model.summary()

    cbs=[EarlyStopping(patience=10,restore_best_weights=True,monitor='val_loss'),
         ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=4,verbose=1),
         ModelCheckpoint('best_model.keras',save_best_only=True,monitor='val_loss')]

    hist = model.fit(Xtr, ytr, epochs=60, batch_size=64, validation_split=0.2,
                     shuffle=True, callbacks=cbs, verbose=1)

    loss, acc, auc = model.evaluate(Xte,yte,verbose=0)
    print(f'Window‑level test  acc={acc:.3f}  auc={auc:.3f}')

    y_prob = model.predict(Xte,verbose=0).squeeze(); y_pred=(y_prob>0.5).astype(int)
    print('\nClassification report:\n',
          classification_report(yte,y_pred,target_names=['Normal','PAF']))
    plot_confusion(confusion_matrix(yte,y_pred))
    plot_history(hist)

    model.save('ecg_paf_detector.keras')
    print('Saved final model → ecg_paf_detector.keras')


if __name__ == '__main__':
    main()