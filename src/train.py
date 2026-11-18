import argparse
import torch
from src.data import make_datasets, make_loader
from src.models import create_model
from src.federated import federated_train
from src.utils import evaluate, save_checkpoint, save_metrics

def main():
    parser = argparse.ArgumentParser(description="Train WBC ViT / FedAvg research model")
    parser.add_argument("--data_dir", default="data/blood_cells")
    parser.add_argument("--mode", choices=["federated", "centralized"], default="federated")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--dp_noise", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, _ = make_datasets(args.data_dir)
    val_loader = make_loader(val_ds, args.batch_size)
    model = create_model("vit").to(device)
    if args.mode == "federated":
        model, history = federated_train(
            model, train_ds, val_loader, device, args.rounds,
            args.local_epochs, args.batch_size, args.lr, args.dp, args.dp_noise
        )
    else:
        loader = make_loader(train_ds, args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        history = []
        for epoch in range(args.epochs):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss_fn(model(x), y).backward()
                optimizer.step()
            m = evaluate(model, val_loader, device)
            history.append({"epoch": epoch+1, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    metrics = evaluate(model, val_loader, device)
    save_checkpoint(model, "models/global_vit.pth",
                    {"classes": ["eosinophil","lymphocyte","monocyte","neutrophil"],
                     "mode": args.mode, "image_size": 32})
    save_metrics({"history": history, "final": metrics, "mode": args.mode})
    print(f"Device: {device}")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
    print("Saved: models/global_vit.pth and models/metrics.json")

if __name__ == "__main__":
    main()
