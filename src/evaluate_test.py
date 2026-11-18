
import torch
import json
from pathlib import Path
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models import CompactViT
from data import CLASSES, build_transforms
from utils import load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = Path("data/test")
MODEL_PATH = Path("models/global_vit.pth")
OUTPUT_PATH = Path("models/test_results.json")


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError("Test dataset folder not found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model checkpoint not found.")

    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=build_transforms(train=False)
    )

    # Convert folder names to lowercase and verify class order.
    found_classes = [name.lower() for name in test_dataset.classes]

    if found_classes != CLASSES:
        raise ValueError(
            f"Expected classes {CLASSES}, found {found_classes}"
        )

    # Convert labels to the model's class order.
    class_mapping = {
        old_index: CLASSES.index(class_name.lower())
        for old_index, class_name in enumerate(test_dataset.classes)
    }

    test_dataset.targets = [
        class_mapping[target] for target in test_dataset.targets
    ]

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    model = CompactViT(num_classes=4, image_size=32)

    load_checkpoint(model, MODEL_PATH, DEVICE)

    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    results = {
        "test_images": len(test_dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nINDEPENDENT TEST RESULTS")
    print("------------------------")
    print("Total test images:", len(test_dataset))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-score: {macro_f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
