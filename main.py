import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import preprocess_data, extract_features
from visualization import *
from model import AudioCNN
from train import Trainer

MODEL_PATH = "saved_model.pth"

if __name__ == "__main__":
    # dataset paths
    metadata_path = "project/ESC-50-master/meta/esc50.csv"
    data_path = "project/ESC-50-master/audio"
    
    load_model = True  # set to False to force training

    # load and preprocess data
    X_train, X_test, y_train, y_test, encoder, file_paths = preprocess_data(
        metadata_path, data_path, return_file_paths=True
    )

    # convert to pytorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(f"training tensor shape: {X_train.shape}, testing tensor shape: {X_test.shape}")

    # select a random file from dataset for visualization
    random_file = random.choice(file_paths)
    print(f"random file chosen: {random_file}")

    # plot original waveform and spectrogram
    plot_waveform(random_file, title="original waveform")
    plot_spectrogram(random_file, title="original spectrogram")

    # apply feature extraction
    processed_features, processed_waveform, sr = extract_features(random_file)

    # plot processed waveform and spectrogram
    if processed_features is not None and processed_waveform is not None:
        plot_processed_waveform(processed_waveform, title="processed waveform")
        plot_processed_spectrogram(processed_waveform, title="processed spectrogram")
    else:
        print("feature extraction failed.")

    # plot dataset class distribution
    plot_class_distribution(y_train, encoder, dataset_type="training set")
    plot_class_distribution(y_test, encoder, dataset_type="test set")

    # initialize model
    num_classes = len(encoder.classes_)
    model = AudioCNN(num_classes)

    # define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # load existing model if available
    if load_model and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("loaded saved model. skipping training.")
    else:
        trainer = Trainer(model, optimizer, criterion, scheduler, epochs=20)
        trainer.train(X_train, y_train)

        # save the trained model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"model saved to {MODEL_PATH}")

    # evaluate model
    trainer = Trainer(model, optimizer, criterion, scheduler, epochs=0)
    y_true_train, y_pred_train, class_acc_train = trainer.evaluate(X_train, y_train, encoder)
    y_true_test, y_pred_test, class_acc_test = trainer.evaluate(X_test, y_test, encoder)

    # plot confusion matrices
    plot_confusion_matrix_clean(y_true_train, y_pred_train, encoder.classes_, dataset_type="training set")
    plot_confusion_matrix_clean(y_true_test, y_pred_test, encoder.classes_, dataset_type="test set")

    # plot per-class accuracy with labels indicating training or testing
    plot_per_class_accuracy(class_acc_train, encoder.classes_, dataset_type="training set")
    plot_per_class_accuracy(class_acc_test, encoder.classes_, dataset_type="test set")

    print("run 'tensorboard --logdir=runs' to visualize training logs.")
