

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

# --------------------------- parameters ----------------------------------
SEGMENT_SECONDS = 10.0
OVERLAP         = 0.5
NORMAL_DIR      = "./Norm data"
PAF_DIR         = "./Paf data"

# ------------------------------------------------------------------------


def discover_records(folder: str) -> List[Path]:
    """Return a sorted list of WFDB base paths (no extension) in *folder*."""
    return sorted(Path(folder).glob("*.hea"))


def load_record(base: Path) -> Tuple[np.ndarray, int]:
    """Load a WFDB record, return (signal[float32], fs)."""
    rec = wfdb.rdrecord(str(base.with_suffix("")))
    sig = rec.p_signal.astype(np.float32)
    if sig.ndim == 2 and sig.shape[1] > 1:
        sig = sig[:, 0]
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)  # ε to avoid /0
    return sig, int(rec.fs)



def slice_windows(sig: np.ndarray, fs: int,
                  seconds: float, overlap: float) -> np.ndarray:
    """Return an array of shape (N, win_len) with safe boundary handling.

    Implements a simple Python loop to avoid the rare off‑by‑one index
    that could appear with the previous vectorised arange() formula.
    """
    win_len = int(seconds * fs)
    step    = int(win_len * (1.0 - overlap))
    if step <= 0:
        step = win_len   # no overlap if overlap>=1

    if len(sig) < win_len:
        return np.empty((0, win_len), dtype=np.float32)

    windows = [
        sig[start:start + win_len]
        for start in range(0, len(sig) - win_len + 1, step)
    ]
    return np.asarray(windows, dtype=np.float32)
    idx = np.arange(win_len)[None, :] + step * np.arange(
        0, len(sig) - win_len + 1, step)[:, None]
    return sig[idx]


def build_model(input_len: int) -> Sequential:
    model = Sequential([
        Conv1D(64, 7, activation='relu', input_shape=(input_len, 1)),
        MaxPooling1D(3),
        Dropout(0.2),

        Conv1D(128, 7, activation='relu'),
        MaxPooling1D(3),
        Dropout(0.2),

        Conv1D(256, 7, activation='relu'),
        MaxPooling1D(3),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def plot_history(hist):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.title('Loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def plot_confusion(cm: np.ndarray):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Normal','PAF'])
    plt.yticks([0,1], ['Normal','PAF'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i,j]),
                     ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.savefig('confusion_matrix.pdf')
    plt.close()


def main():
    # ------------------ discover records ---------------------------------
    normal_headers = discover_records(NORMAL_DIR)
    paf_headers    = discover_records(PAF_DIR)

    if not normal_headers or not paf_headers:
        raise SystemExit("No WFDB headers found – check data folders.")

    print(f"Found {len(normal_headers)} normal and {len(paf_headers)} PAF records.")

    # Reference sampling rate
    ref_sig, ref_fs = load_record(normal_headers[0])
    win_len = int(ref_fs * SEGMENT_SECONDS)
    print(f"Sampling rate: {ref_fs} Hz  ->  window length = {win_len} samples")

    # ------------------ split by record ----------------------------------
    labels_rec = np.array([0]*len(normal_headers) + [1]*len(paf_headers))
    headers_all = np.array(normal_headers + paf_headers)

    h_train, h_test, y_train_rec, y_test_rec = train_test_split(
        headers_all, labels_rec, test_size=0.2, random_state=42,
        stratify=labels_rec)

    # further split train -> train/val inside model.fit(validation_split=0.2)

    # ------------------ slice windows ------------------------------------
    def extract_windows(header_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for h in header_paths:
            sig, fs = load_record(h)
            if fs != ref_fs:
                print(f"Skipping {h.name}: fs={fs} Hz (expected {ref_fs})")
                continue
            win = slice_windows(sig, fs, SEGMENT_SECONDS, OVERLAP)
            xs.append(win)
            ys.extend([1 if h.name[0].lower() == 'p' else 0]*len(win))
        if not xs:
            return np.empty((0,win_len),dtype=np.float32), np.empty((0,),dtype=int)
        return np.concatenate(xs), np.asarray(ys, dtype=int)

    X_train_win, y_train_win = extract_windows(list(h_train))
    X_test_win,  y_test_win  = extract_windows(list(h_test))

    # add channel dimension
    X_train_win = X_train_win[..., None]
    X_test_win  = X_test_win[..., None]

    print(f"Train windows: {X_train_win.shape[0]}  Test windows: {X_test_win.shape[0]}")

    # ------------------ model / training ---------------------------------
    model = build_model(win_len)
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train_win, y_train_win,
        epochs=60,
        batch_size=64,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # ------------------ evaluation ---------------------------------------
    loss, acc, auc = model.evaluate(X_test_win, y_test_win, verbose=0)
    print(f"Window‑level  acc={acc:.3f}  auc={auc:.3f}")

    y_prob = model.predict(X_test_win, verbose=0).squeeze()
    y_pred = (y_prob > 0.5).astype(int)
    print("\nWindow‑level report:\n",
          classification_report(y_test_win, y_pred, target_names=['Normal','PAF']))

    cm = confusion_matrix(y_test_win, y_pred)
    plot_confusion(cm)
    plot_history(history)

    model.save('ecg_paf_detector.keras')
    print("Saved final model → ecg_paf_detector.keras")


def visualize_filters(model, layer_name='conv1d', save_path='filter_visualization.pdf'):
    """Visualize the filters in a specific convolutional layer"""
    # Get the layer
    layer = model.get_layer(name=layer_name)

    # Get the weights
    weights = layer.get_weights()[0]

    # Reshape for visualization
    n_filters = weights.shape[-1]
    filter_length = weights.shape[0]

    # Create a figure
    plt.figure(figsize=(15, n_filters // 4))
    for i in range(min(n_filters, 32)):  # Show max 32 filters
        plt.subplot(8, 4, i + 1)
        plt.plot(weights[:, 0, i])
        plt.title(f'Filter {i + 1}')
        plt.xticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_feature_maps(model, X_sample, layer_names, save_path='feature_maps.pdf'):
    """Generate feature maps for a sample input"""
    # Create models to output feature maps
    feature_models = {name: Model(inputs=model.input,
                                  outputs=model.get_layer(name).output)
                      for name in layer_names}

    # Select a sample (first sample with PAF)
    if X_sample.shape[0] > 0:
        plt.figure(figsize=(15, 10))

        # Plot original signal
        plt.subplot(len(layer_names) + 1, 1, 1)
        plt.plot(X_sample[0, :, 0])
        plt.title('Original ECG Signal')

        # Plot feature maps
        for i, name in enumerate(layer_names):
            features = feature_models[name].predict(X_sample[0:1], verbose=0)
            features = features[0]

            plt.subplot(len(layer_names) + 1, 1, i + 2)

            # If many channels, show average activation
            if features.shape[-1] > 10:
                mean_activation = np.mean(features, axis=-1)
                plt.plot(mean_activation)
                plt.title(f'Mean Activation - {name}')
            else:
                # Show first few channels
                for j in range(min(features.shape[-1], 5)):
                    plt.plot(features[:, j], label=f'Channel {j + 1}')
                plt.legend()
                plt.title(f'Feature Maps - {name}')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def extract_features_to_csv(model, X_data, y_data, layer_name, save_path='features.csv'):
    """Extract features from a specific layer and save to CSV"""
    # Create feature extractor model
    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer(layer_name).output)

    # Extract features
    features = feature_extractor.predict(X_data, verbose=0)

    # Reshape features to 2D
    original_shape = features.shape
    features_2d = features.reshape(features.shape[0], -1)

    # Apply PCA if dimension is too high
    if features_2d.shape[1] > 50:
        pca = PCA(n_components=50)
        features_2d = pca.fit_transform(features_2d)
        print(f"Applied PCA: {original_shape} -> {features_2d.shape}")

        # Create feature names
        feature_names = [f'PCA_Component_{i + 1}' for i in range(features_2d.shape[1])]
    else:
        # Create feature names
        feature_names = [f'Feature_{i + 1}' for i in range(features_2d.shape[1])]

    # Create dataframe
    df = pd.DataFrame(features_2d, columns=feature_names)
    df['Class'] = y_data
    df['Label'] = df['Class'].map({0: 'Normal', 1: 'PAF'})

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Saved features to {save_path}")

    return df


def create_performance_report(model, X_test, y_test, save_path='performance_report.pdf'):
    """Create comprehensive performance report"""
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Create plots
    plt.figure(figsize=(15, 10))

    # ROC curve
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Probability distribution
    plt.subplot(2, 2, 2)
    plt.hist(y_pred_prob[y_test == 0], bins=20, alpha=0.5, label='Normal')
    plt.hist(y_pred_prob[y_test == 1], bins=20, alpha=0.5, label='PAF')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.legend()

    # Extract features from the dense layer
    dense_layer = 'dense'  # Adjust based on your model layer names
    dense_features = Model(inputs=model.input, outputs=model.get_layer(dense_layer).output).predict(X_test, verbose=0)

    # Apply PCA for visualization
    if dense_features.shape[1] > 2:
        pca = PCA(n_components=2)
        dense_features_2d = pca.fit_transform(dense_features)
    else:
        dense_features_2d = dense_features

    # Plot PCA
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(dense_features_2d[:, 0], dense_features_2d[:, 1], c=y_test, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Dense Layer Features')

    # Plot some sample predictions
    plt.subplot(2, 2, 4)
    correct_normal = np.where((y_test == 0) & (y_pred == 0))[0]
    incorrect_normal = np.where((y_test == 0) & (y_pred == 1))[0]
    correct_paf = np.where((y_test == 1) & (y_pred == 1))[0]
    incorrect_paf = np.where((y_test == 1) & (y_pred == 0))[0]

    indices = []
    if len(correct_normal) > 0: indices.append(correct_normal[0])
    if len(incorrect_normal) > 0: indices.append(incorrect_normal[0])
    if len(correct_paf) > 0: indices.append(correct_paf[0])
    if len(incorrect_paf) > 0: indices.append(incorrect_paf[0])

    for i, idx in enumerate(indices[:4]):
        plt.plot(X_test[idx, :, 0] + i * 3, label=f"True:{y_test[idx]}, Pred:{y_pred[idx]}")

    plt.legend()
    plt.title('Sample ECG Signals with Predictions')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Add this to your main() function:
def enhanced_evaluation(model, X_test_win, y_test_win):
    # Visualize filters from first conv layer
    visualize_filters(model, layer_name='conv1d', save_path='cnn_filters.pdf')

    # Get sample data for feature map visualization
    sample_normal = X_test_win[y_test_win == 0][:1]
    sample_paf = X_test_win[y_test_win == 1][:1]

    if len(sample_normal) > 0:
        generate_feature_maps(model, sample_normal,
                              ['conv1d', 'conv1d_1', 'conv1d_2'],
                              save_path='normal_feature_maps.pdf')

    if len(sample_paf) > 0:
        generate_feature_maps(model, sample_paf,
                              ['conv1d', 'conv1d_1', 'conv1d_2'],
                              save_path='paf_feature_maps.pdf')

    # Extract features to CSV
    extract_features_to_csv(model, X_test_win, y_test_win, 'dense', save_path='cnn_features.csv')

    # Create performance report
    create_performance_report(model, X_test_win, y_test_win, save_path='detailed_performance.pdf')


if __name__ == '__main__':
    main()
