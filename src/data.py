from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]
IMAGE_SIZE = 32

def build_transforms(train=False):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)]
    return transforms.Compose(ops)

def make_datasets(data_dir, val_fraction=0.2, seed=42):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")
    base = datasets.ImageFolder(root=str(root), transform=build_transforms(False))
    found = set(base.classes)
    missing = set(CLASSES) - found
    if missing:
        raise ValueError(
            "Missing class folders: " + ", ".join(sorted(missing)) +
            ". Expected: " + ", ".join(CLASSES)
        )
    # Enforce the paper's four-class order consistently.
    if base.classes != CLASSES:
        remap = {base.class_to_idx[name]: i for i, name in enumerate(CLASSES)}
        base.targets = [remap[t] for t in base.targets]
        base.classes = CLASSES
        base.class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    indices = list(range(len(base)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed,
        stratify=base.targets
    )
    train_ds = datasets.ImageFolder(root=str(root), transform=build_transforms(True))
    val_ds = datasets.ImageFolder(root=str(root), transform=build_transforms(False))
    # Match class order and target labels in both datasets.
    for ds in (train_ds, val_ds):
        remap = {ds.class_to_idx[name]: i for i, name in enumerate(CLASSES)}
        ds.targets = [remap[t] for t in ds.targets]
        ds.classes = CLASSES
        ds.class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx), CLASSES

def make_loader(dataset, batch_size=16, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)
