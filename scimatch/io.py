"""Anonymous JSONL records and frozen checkpoint serialization."""
import hashlib
import json
from pathlib import Path
import torch
from .features import FeatureEncoder
from .predictor import CommittedStepMLP
from .calibration import WilsonMap


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(path, model, encoder: FeatureEncoder, calibration: WilsonMap, manifest: dict):
    torch.save({"model": model.state_dict(), "encoder": encoder.state_dict(),
                "calibration": calibration.state_dict(), "manifest": manifest}, path)


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = CommittedStepMLP()
    model.load_state_dict(bundle["model"])
    model.eval()
    enc = FeatureEncoder(**bundle["encoder"])
    cal = WilsonMap.from_dict(bundle["calibration"])
    return model, enc, cal, bundle["manifest"]
