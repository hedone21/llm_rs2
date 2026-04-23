"""Fixture/config loading helpers (models.toml, devices.toml, budgets.toml)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFY_DIR = PROJECT_ROOT / "verify"
FIXTURES_DIR = VERIFY_DIR / "fixtures"
SCENARIOS_DIR = VERIFY_DIR / "scenarios"
RESULTS_DIR = VERIFY_DIR / "results"


def load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_models_toml(path: Path | None = None) -> Dict[str, Any]:
    """Parse fixtures/models.toml.

    Structure:
      [devices.<device_id>.<model_key>]
      model_path = "..."   # absolute or relative to project root
      tokenizer_path = "..."
    """
    p = Path(path) if path else (FIXTURES_DIR / "models.toml")
    return load_toml(p)


def load_device_config(
    devices_toml_path: Path | None = None,
    device_key: str = "host",
) -> Dict[str, Any]:
    p = Path(devices_toml_path) if devices_toml_path else (PROJECT_ROOT / "devices.toml")
    raw = load_toml(p)
    devices = raw.get("devices", {})
    if device_key not in devices:
        raise KeyError(f"Device '{device_key}' not found in {p}")
    cfg = dict(devices[device_key])
    cfg.setdefault("_key", device_key)
    return cfg


def resolve_model_entry(models_toml: Dict[str, Any], device: str, model_key: str) -> Dict[str, Any]:
    """Fetch the {model_path, tokenizer_path, ...} entry for (device, model)."""
    try:
        return dict(models_toml["devices"][device][model_key])
    except KeyError as e:
        raise KeyError(
            f"models.toml has no entry for devices.{device}.{model_key}"
        ) from e


def resolve_tokenizer_path(
    device: str,
    model_key: str,
    project_root: Path | None = None,
    models_toml_path: Path | None = None,
) -> Path:
    """Resolve the LOCAL tokenizer path for (device, model).

    Detokenisation runs host-side regardless of where inference happens, so
    this always returns a local filesystem path. If the models.toml entry
    encodes a relative path, it is resolved against `project_root`.
    """
    models = load_models_toml(models_toml_path)
    entry = resolve_model_entry(models, device, model_key)
    p = Path(entry["tokenizer_path"])
    if not p.is_absolute():
        root = project_root or PROJECT_ROOT
        p = root / p
    return p


def load_prompt(prompt_ref: str) -> str:
    """Resolve a baseline.prompt reference.

    - "fixtures:<name>" → `fixtures/prompts/<name>.txt`
    - absolute/relative path → read verbatim
    """
    if prompt_ref.startswith("fixtures:"):
        name = prompt_ref.split(":", 1)[1]
        path = FIXTURES_DIR / "prompts" / f"{name}.txt"
    else:
        path = Path(prompt_ref)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
