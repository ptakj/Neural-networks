import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


 
# KONFIGURACJA
 

DATA_DIR = Path("./Dataset")
RESULTS_DIR = Path("./training_results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0003
IMAGE_SIZE = 160
PATIENCE = 8

SEED = 42
NUM_WORKERS = 0

CLASS_NAMES = [
    "Armadilles",
    "Bear",
    "Birds",
    "Cow",
    "Crocodile",
    "Deer",
    "Elephant",
    "Goat",
    "Horse",
    "Jaguar",
    "Monkey",
    "Rabbit",
    "Skunk",
    "Tiger",
    "Wild Boar"
]


 
# POWTARZALNOŚĆ WYNIKÓW
 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


 
# GPU
 

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA jest niedostępna! "
        "Sprawdź instalację PyTorch."
    )

device = torch.device("cuda:0")

print("=" * 60)
print("TRENING CNN - NVIDIA GPU")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Urządzenie:", device)


 
# TRANSFORMACJE
 

train_transform = transforms.Compose([
    transforms.Resize((180, 180)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([
    transforms.Resize((180, 180)),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


 
# WCZYTYWANIE DATASETU
 

train_dir = DATA_DIR / "train"
validation_dir = DATA_DIR / "validation"
test_dir = DATA_DIR / "test"

for folder in [train_dir, validation_dir, test_dir]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu: {folder}"
        )


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    validation_dir,
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=validation_transform
)


 
# SPRAWDZENIE KLAS
 

print("\nKlasy znalezione w datasetach:")

print("Train:", train_dataset.classes)
print("Validation:", validation_dataset.classes)
print("Test:", test_dataset.classes)

expected_classes = sorted(CLASS_NAMES)

for name, dataset in [
    ("train", train_dataset),
    ("validation", validation_dataset),
    ("test", test_dataset)
]:
    if dataset.classes != expected_classes:
        raise ValueError(
            f"Niezgodne klasy w {name}!\n"
            f"Oczekiwano: {expected_classes}\n"
            f"Znaleziono: {dataset.classes}"
        )

print("\nLiczba klas:", len(train_dataset.classes))
print("Liczba obrazów train:", len(train_dataset))
print("Liczba obrazów validation:", len(validation_dataset))
print("Liczba obrazów test:", len(test_dataset))


# Zapis mapowania klas

class_mapping = train_dataset.class_to_idx

with open(
    RESULTS_DIR / "class_mapping.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        class_mapping,
        f,
        indent=4,
        ensure_ascii=False
    )


 
# DATALOADERS
 

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


 
# BLOK CNN
 

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)

        )

    def forward(self, x):
        return self.block(x)


 
# MODEL CNN
 

class AnimalCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32),

            ConvBlock(32, 64),

            ConvBlock(64, 128),

            ConvBlock(128, 256),

            ConvBlock(256, 384)

        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(384, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


 
# INICJALIZACJA MODELU
 

model = AnimalCNN(
    num_classes=len(CLASS_NAMES)
).to(device)

print("\n" + "=" * 60)
print("DIAGNOSTYKA MODELU")
print("=" * 60)

print("Model:", next(model.parameters()).device)

print(
    "Liczba parametrów:",
    sum(p.numel() for p in model.parameters())
)

print(
    "Parametry trenowalne:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


 
# TEST GPU
 

print("\nTest wykonywania obliczeń na GPU...")

test_tensor = torch.randn(
    2, 3, IMAGE_SIZE, IMAGE_SIZE
).to(device)

with torch.no_grad():

    test_output = model(test_tensor)

print("Tensor:", test_tensor.device)
print("Wynik modelu:", test_output.device)
print("Kształt wyniku:", test_output.shape)

del test_tensor, test_output

torch.cuda.empty_cache()


 
# LOSS, OPTIMIZER, SCHEDULER
 

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


 
# FUNKCJA SPRAWDZAJĄCA WARTOŚCI
 

def check_finite(tensor, description):

    if not torch.isfinite(tensor).all():

        raise RuntimeError(
            f"Wykryto NaN lub Inf: {description}"
        )


 
# TRENING JEDNEJ EPOKI
 

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        check_finite(images, "obrazy treningowe")

        optimizer.zero_grad(set_to_none=True)

        # FP32 - bez AMP i bez autocast

        outputs = model(images)

        check_finite(outputs, "wyjście modelu podczas treningu")

        loss = criterion(outputs, labels)

        check_finite(loss, "loss podczas treningu")

        # Zwykły backpropagation FP32

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        if batch_idx == 0:

            print("\nDiagnostyka batcha:")
            print("Dane:", images.device)
            print("Etykiety:", labels.device)
            print("Model:", next(model.parameters()).device)
            print("Loss:", loss.item())

    epoch_loss = total_loss / total

    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


 
# WALIDACJA
 

def validate_one_epoch(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            check_finite(images, "obrazy walidacyjne")

            outputs = model(images)

            check_finite(
                outputs,
                "wyjście modelu podczas walidacji"
            )

            loss = criterion(outputs, labels)

            check_finite(
                loss,
                f"loss walidacyjny, batch {batch_idx}"
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    epoch_loss = total_loss / total

    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


 
# GŁÓWNA PĘTLA TRENINGOWA
 

history = {
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": [],
    "learning_rate": []
}

best_validation_loss = float("inf")
epochs_without_improvement = 0

best_model_path = RESULTS_DIR / "best_model.pth"

print("\n" + "=" * 60)
print("ROZPOCZYNAM TRENING NA GPU - FP32")
print("=" * 60)


for epoch in range(EPOCHS):

    print(f"\nEpoka {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    validation_loss, validation_accuracy = validate_one_epoch(
        model,
        validation_loader,
        criterion
    )

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)

    history["train_accuracy"].append(train_accuracy)

    history["validation_loss"].append(validation_loss)

    history["validation_accuracy"].append(validation_accuracy)

    history["learning_rate"].append(current_lr)

    print(
        f"\nTrain -> Loss: {train_loss:.4f} "
        f"| Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation -> Loss: {validation_loss:.4f} "
        f"| Accuracy: {validation_accuracy:.2f}%"
    )

    print(f"Learning rate: {current_lr:.7f}")

    print(
        f"GPU RAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB "
        f"zajęte, "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB "
        f"zarezerwowane"
    )

    # Zapis najlepszego modelu

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": validation_loss,
                "class_mapping": class_mapping
            },
            best_model_path
        )

        print("Zapisano nowy najlepszy model!")

    else:

        epochs_without_improvement += 1

        print(
            "Brak poprawy:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    # Early stopping

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping - zakończono trening.")

        break


 
# ZAPIS HISTORII TRENINGU
 

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False
)


 
# WYKRESY TRENINGU
 

plt.figure(figsize=(10, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.title("Loss podczas treningu")

plt.legend()
plt.grid()

plt.savefig(
    RESULTS_DIR / "loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["validation_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoka")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy podczas treningu")

plt.legend()
plt.grid()

plt.savefig(
    RESULTS_DIR / "accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


 
# EWALUACJA NA ZBIORZE TESTOWYM
 

print("\n" + "=" * 60)
print("EWALUACJA NA ZBIORZE TESTOWYM")
print("=" * 60)


checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

test_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        check_finite(outputs, "wyjście modelu na zbiorze testowym")

        loss = criterion(outputs, labels)

        check_finite(loss, "loss testowy")

        batch_size = images.size(0)

        test_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_loss /= total

test_accuracy = 100.0 * correct / total

print(f"\nTest Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.2f}%")


 
# CLASSIFICATION REPORT
 

report = classification_report(
    all_labels,
    all_predictions,
    target_names=train_dataset.classes,
    digits=4,
    zero_division=0
)

print("\nRaport klasyfikacji:")
print(report)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


 
# CONFUSION MATRIX
 

cm = confusion_matrix(
    all_labels,
    all_predictions
)

cm_df = pd.DataFrame(
    cm,
    index=train_dataset.classes,
    columns=train_dataset.classes
)

cm_df.to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)

fig, ax = plt.subplots(figsize=(13, 11))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_dataset.classes
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=True
)

plt.title("Confusion Matrix - Test")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()


 
# PODSUMOWANIE
 

summary = {
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0),
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "epochs_completed": len(history["train_loss"]),
    "best_validation_loss": float(best_validation_loss),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "number_of_classes": len(CLASS_NAMES),
    "number_of_parameters": sum(
        p.numel() for p in model.parameters()
    )
}

with open(
    RESULTS_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 60)
print("TRENING ZAKOŃCZONY")
print("=" * 60)

print(f"Najlepszy Validation Loss: {best_validation_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.2f}%")

print("\nWyniki zapisano w:", RESULTS_DIR.resolve())

print("\nPliki:")
print("- best_model.pth")
print("- training_history.csv")
print("- training_summary.json")
print("- classification_report.txt")
print("- confusion_matrix.csv")
print("- confusion_matrix.png")
print("- loss_curve.png")
print("- accuracy_curve.png")
print("- class_mapping.json")