import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

writer = SummaryWriter()


class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model, optimizer, criterion, scheduler, epochs=20):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def train(self, X_train, y_train):
        """Trains the model for a set number of epochs."""
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # compute accuracy
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_train).sum().item() / y_train.size(0)
            self.train_accuracy_history.append(acc)

            writer.add_scalar("accuracy/train", acc, epoch)

            print(f"epoch [{epoch+1}/{self.epochs}], training accuracy: {acc:.4f}")

    def evaluate(self, X_test, y_test, encoder):
        """Evaluates the model and prints accuracy and classification metrics."""
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
            acc = accuracy_score(y_test.numpy(), predicted.numpy())
            self.test_accuracy_history.append(acc)

            # compute per-class accuracy
            class_acc = confusion_matrix(y_test.numpy(), predicted.numpy(), normalize="true").diagonal()
            writer.add_scalar("accuracy/test", acc)

            print(f"test accuracy: {acc:.4f}\n")
            print("classification report:\n", classification_report(y_test.numpy(), predicted.numpy(), target_names=encoder.classes_))
            print("\nper-class accuracy:")
            for i, label in enumerate(encoder.classes_):
                print(f"{label}: {class_acc[i]:.2f}")

            return y_test.numpy(), predicted.numpy(), class_acc
