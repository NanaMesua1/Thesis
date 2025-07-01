#!/usr/bin/env python3
import time, smbus
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import tensorflow as tf

# === USER CONFIG ===
I2C_BUS      = 1
ADDR         = 0x48
FS           = 128        # Hz
DURATION     = 600        # seconds
PGA_FS       = 4.096      # ±4.096 V
MODEL_PATH   = "/home/pi/Thesis/ecg_paf_detector.keras"
WINDOW_SEC   = 10         # CNN input length (s)
THRESHOLD    = 0.5        # PAF/non-PAF cutoff
# ECG-plot zoom
ZOOM_START   = 9.0        # seconds into the record
ZOOM_WINDOW  = 600        # length of zoom (s)
# ===================

def read_ecg():
    bus = smbus.SMBus(I2C_BUS)
    n   = FS * DURATION
    out = np.zeros(n)
    # Configure ADS1115: AIN0 vs GND, cont, ±4.096 V, 128 SPS → 0x42 0x83
    bus.write_i2c_block_data(ADDR, 0x01, [0x42, 0x83])
    time.sleep(0.1)
    dt = 1.0/FS
    for i in range(n):
        msb, lsb = bus.read_i2c_block_data(ADDR, 0x00, 2)
        raw = (msb << 8) | lsb
        if raw & 0x8000:
            raw -= (1 << 16)
        out[i] = raw * (PGA_FS/32768.0)
        time.sleep(dt)
    return out

def preprocess(ecg):
    ecg = ecg - np.mean(ecg)  # DC remove
    b,a = butter(4, [0.5/(FS/2), 40/(FS/2)], btype='band')
    ecg = filtfilt(b, a, ecg)
    return ecg * 1000.0        # to millivolts

def windowize(ecg):
    L = FS * WINDOW_SEC
    nwin = len(ecg)//L
    return ecg[:nwin*L].reshape(nwin, L)

def plot_ecg(ecg_mV):
    t = np.arange(len(ecg_mV))/FS
    # Zoom in on short window
    mask = (t >= ZOOM_START) & (t < ZOOM_START + ZOOM_WINDOW)
    plt.figure(figsize=(6,3))
    plt.plot(t[mask], ecg_mV[mask], linewidth=1)
    plt.title(f"Filtered ECG ({ZOOM_WINDOW}s @ {FS}Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_cnn(windows):
    model = tf.keras.models.load_model(MODEL_PATH)
    x = windows[..., np.newaxis]  # (batch, length, 1)
    preds = model.predict(x, batch_size=4, verbose=0)
    # binary‐sigmoid → shape (n,1)
    scores = preds.ravel()
    return scores

def main():
    # 1) Acquire and preprocess
    raw = read_ecg()
    ecg = preprocess(raw)
    # 2) Display an ECG plot
    plot_ecg(ecg)
    # 3) Windowize & classify
    wins  = windowize(ecg)
    scores = run_cnn(wins)
    # 4) Decision
    if np.any(scores > THRESHOLD):
        print("✨ PAF detected!")
    else:
        print("✅ No PAF detected.")
    for i,s in enumerate(scores):
        print(f"Window {i}: score={s:.3f}")

if __name__=="__main__":
    main()
