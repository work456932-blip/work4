"""Base committed-step predictor; labels refer to failures unresolved at commit."""
from dataclasses import dataclass
import copy
import random
import numpy as np
import torch
from torch import nn


class CommittedStepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(23, 64), nn.GELU(), nn.Dropout(0.10),
                                 nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.10),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits; sigmoid only at inference


@dataclass(frozen=True)
class FitConfig:
    seed: int = 3407
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    validation_fraction: float = 0.10


def fit_predictor(x_fit, y_fit, config: FitConfig = FitConfig()):
    """Split Dfit internally by task group upstream, then train on remaining rows.

    Pass already disjoint internal train/validation matrices to `fit_with_validation`
    when multiple prefixes share a task. This convenience function is only for
    data where each row represents a distinct task group.
    """
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(len(y_fit))
    nval = max(1, round(len(idx) * config.validation_fraction))
    return fit_with_validation(np.asarray(x_fit)[idx[nval:]], np.asarray(y_fit)[idx[nval:]],
                               np.asarray(x_fit)[idx[:nval]], np.asarray(y_fit)[idx[:nval]], config)


def fit_with_validation(x_train, y_train, x_val, y_val, config: FitConfig = FitConfig()):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    model = CommittedStepMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    ty = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    vx = torch.as_tensor(np.asarray(x_val, dtype=np.float32))
    vy = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
    if not len(tx) or not len(vx) or len(tx) != len(ty) or len(vx) != len(vy):
        raise ValueError("invalid internal Dfit split")
    best, best_nll, stale = None, float("inf"), 0
    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(len(tx))
        for batch in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tx[batch]), ty[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            nll = float(loss_fn(model(vx), vy))
        if nll < best_nll:
            best_nll, best, stale = nll, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best)
    model.eval()
    return model, {"validation_nll": best_nll, "epochs": epoch + 1}


@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(np.asarray(x, dtype=np.float32)))).cpu().numpy()
