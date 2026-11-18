import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred,
                    target_names=["eosinophil", "lymphocyte", "monocyte", "neutrophil"],
                    output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1,2,3]).tolist()
    }

def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata or {}}, path)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint.get("metadata", {})

def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
