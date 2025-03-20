import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_waveform(file_path, title="waveform"):
    """plots the waveform of an audio file"""
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.show()


def plot_spectrogram(file_path, title="spectrogram"):
    """plots the spectrogram of an audio file"""
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()


def plot_processed_waveform(y, sr=22050, title="processed waveform"):
    """plots the processed waveform after feature extraction"""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.show()


def plot_processed_spectrogram(y, sr=22050, title="processed spectrogram"):
    """plots the processed spectrogram after feature extraction"""
    S_DB = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S_DB, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()


def plot_class_distribution(y, encoder, dataset_type="dataset"):
    """plots the distribution of classes in a dataset"""
    labels, counts = np.unique(encoder.inverse_transform(y), return_counts=True)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("categories")
    plt.ylabel("count")
    plt.title(f"class distribution in {dataset_type}")
    plt.show()


def plot_accuracy(train_accuracy, test_accuracy):
    """plots training and test accuracy trends"""
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(train_accuracy) + 1),
        train_accuracy,
        marker="o",
        label="train accuracy",
    )
    plt.plot(
        range(1, len(test_accuracy) + 1),
        test_accuracy,
        marker="o",
        linestyle="dashed",
        label="test accuracy",
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("training vs test accuracy trend")
    plt.legend()
    plt.grid()
    plt.show()


def plot_per_class_accuracy(class_acc, class_names, dataset_type="train"):
    """plots per-class accuracy for training or testing"""
    plt.figure(figsize=(12, 5))
    plt.bar(class_names, class_acc)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("class")
    plt.ylabel("accuracy")
    plt.title(f"per-class accuracy ({dataset_type})")
    plt.show()


def plot_confusion_matrix_clean(y_true, y_pred, class_names, dataset_type="dataset"):
    """plots a clean confusion matrix with just colors"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    # remove numbers, show only colors
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot=False,
    )

    plt.xlabel("predicted labels", fontsize=12)
    plt.ylabel("actual labels", fontsize=12)
    plt.title(f"confusion matrix ({dataset_type})", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
