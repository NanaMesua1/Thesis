#!/usr/bin/env python3
import time, smbus
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === USER CONFIG ===
I2C_BUS  = 1
ADDR     = 0x48
FS       = 128       # Hz
DURATION = 600       # seconds
PGA_FS   = 4.096     # ±4.096 V
# Zoom window parameters:
start_s = 9.0        # seconds into the record
win_s   = 1.2        # length of zoom window (s)
# ===================

# Prepare I2C & buffers
bus   = smbus.SMBus(I2C_BUS)
n     = int(FS * DURATION)
data  = np.zeros(n)
delay = 1.0 / FS

# 1) Configure ADS1115 for single-ended AIN0 vs GND, cont, ±4.096 V, 128 SPS
#    → CONFIG = 0x42 0x83
bus.write_i2c_block_data(ADDR, 0x01, [0x42, 0x83])
time.sleep(0.1)

# 2) Read conversion register in a loop
for i in range(n):
    msb, lsb = bus.read_i2c_block_data(ADDR, 0x00, 2)
    raw = (msb << 8) | lsb
    if raw & 0x8000:
        raw -= (1 << 16)
    data[i] = raw * (PGA_FS / 32768.0)
    time.sleep(delay)

# --- DIGITAL PROCESSING AND PLOTTING ---

# A) Remove DC offset
data_dc = data - np.mean(data)

# B) 0.5–40 Hz Butterworth band-pass
low, high = 0.5, 40.0
nyq        = 0.5 * FS
b, a       = butter(4, [low/nyq, high/nyq], btype='band')
ecg_bp     = filtfilt(b, a, data_dc)

# C) Convert to millivolts
ecg_mV = ecg_bp * 1000.0

# D) Extract zoom window
t = np.arange(n) / FS
mask = (t >= start_s) & (t < start_s + win_s)
t_win   = t[mask]
ecg_win = ecg_mV[mask]

# E) Plot clean ECG
plt.figure(figsize=(6,3), dpi=100)
plt.plot(t_win, ecg_win, linewidth=1)
plt.title(f"ECG ({win_s}s @ {FS} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.tight_layout()
plt.show()
