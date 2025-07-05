from pathlib import Path

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall
from tqdm import tqdm

from data_loader import get_balanced_tensor_datasets
from model import TinyConvNet


def save_model(model: nn.Module, path: str = "model.pth") -> None:
    """
    Saves the model state dictionary to the specified path.
    :param model: nn.Module - the model to save.
    :param path: str - the path where the model will be saved.
    :return: None
    """
    Path(path).parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
        num_epochs: int = 10, lr: float = 1e-3, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    model = model.to(device)

    loss_fn = CrossEntropyLoss(weight=torch.tensor([0.33, 0.67], device=device))  # Adjust weights for class imbalance (inverse of class distribution)
    optimizer = Adam(model.parameters(), lr=lr)
    acc = Accuracy(task="multiclass", num_classes=2).to(device)
    auroc = AUROC(task="multiclass", num_classes=2).to(device)
    precision = Precision(task="multiclass", num_classes=2).to(device)
    recall = Recall(task="multiclass", num_classes=2).to(device)

    metrics = [acc, auroc, precision, recall]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        auroc.reset()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                auroc(preds, labels)
            total_loss += loss.item()

        print(f"Train loss: {total_loss/len(train_loader):.4f} AUROC: {auroc.compute().item():.4f}")

        # Validation
        model.eval()
        for metric in metrics:
            metric.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                for metric in metrics:
                    metric(preds, labels)
        val_acc = acc.compute().item()
        val_auroc = auroc.compute().item()
        val_precision = precision.compute().item()
        val_recall = recall.compute().item()

        print(f"\nValidation - Accuracy: {val_acc:.4f}, AUROC: {val_auroc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    print("Evaluating on Test Dataset")
    for metric in metrics:
        metric.reset()
    with torch.no_grad():
        lbl = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            lbl += labels.sum().item()
            preds = model(images)
            for metric in metrics:
                metric(preds, labels)

    print(f"Test - Accuracy: {acc.compute().item():.4f}, AUROC: {auroc.compute().item():.4f}, Precision: {precision.compute().item():.4f}, Recall: {recall.compute().item():.4f}")
    print(f"Total positive labels in test set: {lbl}")
    return model



if __name__ == "__main__":
    v = "tinyv2.1"  # set to the desired model version to train

    model = TinyConvNet(input_channels=1, num_classes=2)
    train_loader, val_loader, test_loader = get_balanced_tensor_datasets(
        root_dir="down_data", batch_size=64, val_size=0.2, test_size=0.1, flip_augment=True
    )

    trained_model = train_model(model, train_loader, val_loader, test_loader, num_epochs=50)
    save_model(trained_model, f"models/model_{v}.pth")
