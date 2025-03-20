import os
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_metadata(metadata_path):
    """loads the metadata csv file"""
    return pd.read_csv(metadata_path)


def extract_features(file_path):
    """extracts a mel spectrogram from an audio file for cnn input"""
    try:
        y, sr = librosa.load(file_path, sr=None)

        # apply noise reduction
        y = nr.reduce_noise(y=y, sr=sr)

        # compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # convert to db scale

        # ensure consistent shape (128, 431)
        if mel_spec_db.shape[1] < 431:
            pad_width = 431 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
        elif mel_spec_db.shape[1] > 431:
            mel_spec_db = mel_spec_db[:, :431]  # truncate if too long

        return mel_spec_db, y, sr  # return the processed spectrogram

    except Exception as e:
        print(f"error processing {file_path}: {e}")
        return None, None, None


def load_data(df, data_path, return_file_paths=False):
    """loads audio data and extracts spectrograms for cnn"""
    X, y, file_paths = [], [], []

    for _, row in df.iterrows():
        file_path = os.path.join(data_path, row["filename"])

        if not os.path.exists(file_path):
            print(f"file not found: {file_path}")
            continue

        features, waveform, sr = extract_features(file_path)  # get 2d spectrograms

        if features is not None:
            X.append(features)  # no need to flatten for cnn
            y.append(row["category"])
            if return_file_paths:
                file_paths.append(file_path)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"final dataset size: X={X.shape}, y={y.shape}")  # should be (samples, 128, 431)

    if return_file_paths:
        return X, y, file_paths
    return X, y


def preprocess_data(metadata_path, data_path, return_file_paths=False):
    """prepares the dataset for training with cnn input"""
    df = load_metadata(metadata_path)
    X, y, file_paths = load_data(df, data_path, return_file_paths=True)  # return file paths

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # add channel dimension for cnn (batch, 1, height, width)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    print(f"final training shape: {X_train.shape}, final testing shape: {X_test.shape}")

    if return_file_paths:
        return X_train, X_test, y_train, y_test, encoder, file_paths
    return X_train, X_test, y_train, y_test, encoder
