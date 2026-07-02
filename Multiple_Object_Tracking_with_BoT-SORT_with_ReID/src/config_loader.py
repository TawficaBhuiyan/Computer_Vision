"""Load config.yaml and expose it with dotted access (cfg.model.weights)."""
import yaml
from types import SimpleNamespace


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str = "config/config.yaml") -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)
