# #!/usr/bin/env python3
# import time, smbus
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
# import tensorflow as tf

# # === USER CONFIG ===
# I2C_BUS      = 1
# ADDR         = 0x48
# FS           = 128        # Hz
# DURATION     = 600        # seconds
# PGA_FS       = 4.096      # ±4.096 V
# MODEL_PATH   = "/home/pi/Thesis/ecg_paf_detector.keras"
# WINDOW_SEC   = 10         # CNN input length (s)
# THRESHOLD    = 0.5        # PAF/non-PAF cutoff
# # ECG-plot zoom
# ZOOM_START   = 9.0        # seconds into the record
# ZOOM_WINDOW  = 600        # length of zoom (s)
# # ===================

# def read_ecg():
#     bus = smbus.SMBus(I2C_BUS)
#     n   = FS * DURATION
#     out = np.zeros(n)
#     # Configure ADS1115: AIN0 vs GND, cont, ±4.096 V, 128 SPS → 0x42 0x83
#     bus.write_i2c_block_data(ADDR, 0x01, [0x42, 0x83])
#     time.sleep(0.1)
#     dt = 1.0/FS
#     for i in range(n):
#         msb, lsb = bus.read_i2c_block_data(ADDR, 0x00, 2)
#         raw = (msb << 8) | lsb
#         if raw & 0x8000:
#             raw -= (1 << 16)
#         out[i] = raw * (PGA_FS/32768.0)
#         time.sleep(dt)
#     return out

# def preprocess(ecg):
#     ecg = ecg - np.mean(ecg)  # DC remove
#     b,a = butter(4, [0.5/(FS/2), 40/(FS/2)], btype='band')
#     ecg = filtfilt(b, a, ecg)
#     return ecg * 1000.0        # to millivolts

# def windowize(ecg):
#     L = FS * WINDOW_SEC
#     nwin = len(ecg)//L
#     return ecg[:nwin*L].reshape(nwin, L)

# def plot_ecg(ecg_mV):
#     t = np.arange(len(ecg_mV))/FS
#     # Zoom in on short window
#     mask = (t >= ZOOM_START) & (t < ZOOM_START + ZOOM_WINDOW)
#     plt.figure(figsize=(6,3))
#     plt.plot(t[mask], ecg_mV[mask], linewidth=1)
#     plt.title(f"Filtered ECG ({ZOOM_WINDOW}s @ {FS}Hz)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Voltage (mV)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def run_cnn(windows):
#     model = tf.keras.models.load_model(MODEL_PATH)
#     x = windows[..., np.newaxis]  # (batch, length, 1)
#     preds = model.predict(x, batch_size=4, verbose=0)
#     # binary‐sigmoid → shape (n,1)
#     scores = preds.ravel()
#     return scores

# def main():
#     # 1) Acquire and preprocess
#     raw = read_ecg()
#     ecg = preprocess(raw)
#     # 2) Display an ECG plot
#     plot_ecg(ecg)
#     # 3) Windowize & classify
#     wins  = windowize(ecg)
#     scores = run_cnn(wins)
#     # 4) Decision
#     if np.any(scores > THRESHOLD):
#         print("✨ PAF detected!")
#     else:
#         print("✅ No PAF detected.")
#     for i,s in enumerate(scores):
#         print(f"Window {i}: score={s:.3f}")

# if __name__=="__main__":
#     main()


# ---------------------------------------------------------------------------------------------------------------------

#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  ecg_paf_monitor.py            (Raspberry Pi 3  •  ADS1115  •  TFLite CNN)
#
#  • Continuously acquires single-lead ECG via ADS1115
#  • Streams the raw counts to disk *and* classifies 10-s windows on-device
#  • Stores each session in WFDB format  (<name>_YYYYMMDD_HHMMSS.{dat,hea,qrs})
# ---------------------------------------------------------------------------
import argparse, datetime, time, os, sys, random
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
import smbus2                                # slightly faster than smbus
import tensorflow as tf                      # only the TFLite interpreter is used
import wfdb

# ────────────────────────────── USER CONSTANTS ─────────────────────────────
FS            = 128          # ADS1115 sample-rate (Hz)
PGA_FS        = 4.096        # ±4.096 V → 1 LSB = 125 µV
WINDOW_SEC    = 10           # CNN input length
THRESHOLD     = 0.50         # sigmoid score → PAF if any win > THRESHOLD
ADS_ADDR      = 0x48
I2C_BUS       = 1
MODEL_PATH    = "/home/pi/Thesis/ecg_paf_detector.tflite"
SAVE_DIR      = Path("/home/pi/Records")     # will be created if missing
# ───────────────────────────────────────────────────────────────────────────

### 1 ─────────────────────────────────  ADC helpers  ──────────────────────
def init_ads1115(bus: smbus2.SMBus) -> None:
    """
    Continuous conversion • AIN0 • 128 SPS • PGA ±4.096 V
    Config word = 0b1000 0010 1000 0011  ->  0x82 0x83
    """
    bus.write_i2c_block_data(ADS_ADDR, 0x01, [0x82, 0x83])  # CFG register

def read_samples(n: int) -> np.ndarray:
    """Read *n* samples (blocking) from ADS1115."""
    bus  = smbus2.SMBus(I2C_BUS)
    init_ads1115(bus)
    out  = np.empty(n, dtype=np.float32)
    tick = 1.0 / FS
    for i in range(n):
        msb, lsb = bus.read_i2c_block_data(ADS_ADDR, 0x00, 2)
        raw = (msb << 8) | lsb
        if raw & 0x8000:                # two’s-complement to signed
            raw -= 1 << 16
        out[i] = raw * (PGA_FS / 32768.0)
        time.sleep(tick)
    return out

### 2 ─────────────────────────────  DSP blocks  ───────────────────────────
BP_B, BP_A = butter(4, [0.5/(FS/2), 40/(FS/2)], btype='band')

def preprocess(ecg_volts: np.ndarray) -> np.ndarray:
    """DC-remove, band-pass 0.5-40 Hz, → millivolts."""
    ecg = ecg_volts - np.mean(ecg_volts)
    ecg = filtfilt(BP_B, BP_A, ecg)
    return ecg * 1000.0                   # mV

def windowize(ecg_mV: np.ndarray) -> np.ndarray:
    L = FS * WINDOW_SEC
    n = len(ecg_mV) // L
    return ecg_mV[:n*L].reshape(n, L).astype(np.float32)

### 3 ─────────────────────────────  TFLite runner  ─────────────────────────
def tflite_predict(windows: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp_index  = interpreter.get_input_details()[0]['index']
    out_index  = interpreter.get_output_details()[0]['index']

    scores = np.empty((windows.shape[0],), dtype=np.float32)
    for i, win in enumerate(windows):
        interpreter.set_tensor(inp_index, win.reshape(1, -1, 1))
        interpreter.invoke()
        scores[i] = interpreter.get_tensor(out_index).ravel()[0]
    return scores

### 4 ─────────────────────────────  WFDB writer  ──────────────────────────
def save_wfdb(record_dir: Path,
              patient: str,
              raw_counts: np.ndarray) -> None:
    """
    Stores <name>.hea / <name>.dat / <name>.qrs inside *record_dir*.

    • .dat : int16 raw ADC counts (gain = 125 µV)
    • .hea : minimal WFDB header
    • .qrs : empty stub that later QRS detectors can overwrite
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    rec_name = record_dir.name                 # WFDB uses folder name as prefix
    # convert counts to µV so wfdb sees physiologically scaled signal
    signal_uV = (raw_counts * 1e6 * PGA_FS / 32768.).astype(np.int16)  # µV → int16
    wfdb.wrsamp(
        rec_name,
        fs     = FS,
        units  = ['uV'],
        sig_name=['ECG'],
        fmt    = ['16'],
        p_signal = signal_uV.reshape(-1, 1) / 1e6   # back to floating volts for wrsamp
    )
    # write an empty QRS annotation placeholder
    (record_dir / f"{rec_name}.qrs").write_text("# empty QRS annotation file\n")

### 5 ───────────────────────────────  main()  ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Real-time PAF monitor + WFDB recorder (ADS1115 → TFLite)")
    ap.add_argument("--name", required=True,
                    help="Patient's name  (for file naming)")
    ap.add_argument("--minutes", type=float, default=10,
                    help="Recording length in minutes (default 10)")
    args = ap.parse_args()

    n_samples = int(FS * 60 * args.minutes)
    print(f"[i] Recording {args.minutes:.1f} min  ({n_samples} samples)…")
    raw_volts = read_samples(n_samples)        # blocking acquisition
    ecg_mV    = preprocess(raw_volts)
    wins      = windowize(ecg_mV)
    print(f"[i] Analysing {len(wins)} windows with CNN …")

    scores = tflite_predict(wins)
    paf_flag = bool(np.any(scores > THRESHOLD))
    print("\n──────────  RESULTS  ──────────")
    for i, s in enumerate(scores):
        print(f" Window {i:<2}:  {s:5.3f}")
    print("➜  PAF DETECTED!" if paf_flag else "➜  No PAF evidence.")

    # WFDB storage
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = SAVE_DIR / f"{args.name}_{ts}"
    save_wfdb(dest, args.name, raw_volts * 32768.0 / PGA_FS)   # back to counts
    print(f"[✓] Recording saved   →  {dest}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[!] Interrupted – partial data not saved.")





