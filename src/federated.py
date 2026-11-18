import copy
import random
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from src.utils import evaluate

def split_clients(dataset, num_clients=3, seed=42):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    chunks = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, chunk) for chunk in chunks if chunk]

def train_local(global_model, dataset, device, epochs=1, batch_size=16,
                learning_rate=5e-4, use_dp=False, dp_noise=1.0):
    model = copy.deepcopy(global_model).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = None
    if use_dp:
        try:
            from opacus import PrivacyEngine
        except ImportError as exc:
            raise RuntimeError("DP mode needs Opacus. Install with: pip install opacus") from exc
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=dp_noise, max_grad_norm=1.0
        )
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    if privacy_engine is not None:
        model = model._module
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, len(dataset)

def fedavg(state_dicts, sample_counts):
    total = sum(sample_counts)
    if total == 0:
        raise ValueError("No client training samples were found.")
    result = {}
    for key in state_dicts[0]:
        if torch.is_floating_point(state_dicts[0][key]):
            result[key] = sum(sd[key] * (n / total)
                              for sd, n in zip(state_dicts, sample_counts))
        else:
            result[key] = state_dicts[0][key]
    return result

def federated_train(model, train_dataset, val_loader, device, rounds=5,
                    local_epochs=1, batch_size=16, learning_rate=5e-4,
                    use_dp=False, dp_noise=1.0, progress_callback=None):
    clients = split_clients(train_dataset, 3)
    history = []
    for rnd in range(rounds):
        local_states, counts = [], []
        for client_data in clients:
            state, count = train_local(
                model, client_data, device, epochs=local_epochs,
                batch_size=batch_size, learning_rate=learning_rate,
                use_dp=use_dp, dp_noise=dp_noise
            )
            local_states.append(state)
            counts.append(count)
        model.load_state_dict(fedavg(local_states, counts))
        metrics = evaluate(model, val_loader, device)
        history.append({"round": rnd + 1, "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"]})
        if progress_callback:
            progress_callback(rnd + 1, rounds, metrics)
    return model, history
