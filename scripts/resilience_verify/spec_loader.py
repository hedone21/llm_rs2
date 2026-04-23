"""YAML scenario spec loader.

Phase 1 에서는 Throttle smoke test 1개만 지원하지만, 향후 phase 를 위해 구조를
갖춘 dataclass 로 파싱한다. 현재 사용되지 않는 필드는 Optional 로 허용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BaselineCfg:
    prompt: str                      # "fixtures:medium_qa" or absolute path
    decode_tokens: int = 64
    prefill_tokens_min: int = 0
    seed: int = 42
    greedy: bool = True
    temperature: float = 0.0
    backend: str = "auto"
    prefill_chunk_size: int = 64
    extra_args: List[str] = field(default_factory=list)


@dataclass
class ActionCfg:
    enable_resilience: bool = True
    # layer=engine_cmd 단일 명령 모드 (Phase 1)
    mock_manager_command: Optional[str] = None
    mock_manager_params: Dict[str, Any] = field(default_factory=dict)
    # layer=engine_cmd scenario 모드 (Phase 3)
    mock_manager_commands: Optional[List[Dict[str, Any]]] = None
    # layer=signal (Phase 2)
    injection_schedule: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExpectedCfg:
    functional: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    accuracy: Dict[str, Any] = field(default_factory=dict)
    # v2 — crash / silent-truncation hard gate. Applies regardless of
    # pass_criteria. Defaults in assertions.verify_crash_and_progress.
    crash_and_progress: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    id: str
    description: str
    layer: str                       # "signal" | "engine_cmd"
    devices: List[str]
    models: List[str]
    baseline: BaselineCfg
    action: ActionCfg
    expected: ExpectedCfg
    pass_criteria: str = "all"
    path: Optional[Path] = None


def load_scenario(path: Path) -> ScenarioSpec:
    """Load a single YAML scenario into a ScenarioSpec."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario {path} is not a mapping")

    baseline_raw = raw.get("baseline", {}) or {}
    baseline = BaselineCfg(
        prompt=baseline_raw.get("prompt", "fixtures:medium_qa"),
        decode_tokens=int(baseline_raw.get("decode_tokens", 64)),
        prefill_tokens_min=int(baseline_raw.get("prefill_tokens_min", 0)),
        seed=int(baseline_raw.get("seed", 42)),
        greedy=bool(baseline_raw.get("greedy", True)),
        temperature=float(baseline_raw.get("temperature", 0.0)),
        backend=str(baseline_raw.get("backend", "auto")),
        prefill_chunk_size=int(baseline_raw.get("prefill_chunk_size", 64)),
        extra_args=list(baseline_raw.get("extra_args", []) or []),
    )

    action_raw = raw.get("action", {}) or {}
    action = ActionCfg(
        enable_resilience=bool(action_raw.get("enable_resilience", True)),
        mock_manager_command=action_raw.get("mock_manager_command"),
        mock_manager_params=dict(action_raw.get("mock_manager_params", {}) or {}),
        mock_manager_commands=action_raw.get("mock_manager_commands"),
        injection_schedule=action_raw.get("injection_schedule"),
    )

    expected_raw = raw.get("expected", {}) or {}
    expected = ExpectedCfg(
        functional=dict(expected_raw.get("functional", {}) or {}),
        performance=dict(expected_raw.get("performance", {}) or {}),
        accuracy=dict(expected_raw.get("accuracy", {}) or {}),
        crash_and_progress=dict(expected_raw.get("crash_and_progress", {}) or {}),
    )

    spec = ScenarioSpec(
        id=str(raw["id"]),
        description=str(raw.get("description", "")),
        layer=str(raw.get("layer", "engine_cmd")),
        devices=list(raw.get("devices", [])),
        models=list(raw.get("models", [])),
        baseline=baseline,
        action=action,
        expected=expected,
        pass_criteria=str(raw.get("pass_criteria", "all")),
        path=Path(path),
    )
    return spec


def discover_scenarios(scenarios_dir: Path) -> List[ScenarioSpec]:
    """Scan a directory for .yaml scenarios (excluding _schema.yaml)."""
    out: List[ScenarioSpec] = []
    for p in sorted(scenarios_dir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        out.append(load_scenario(p))
    return out


def filter_scenarios(scenarios: List[ScenarioSpec], substr: Optional[str]) -> List[ScenarioSpec]:
    if not substr:
        return scenarios
    needle = substr.lower().strip()
    if needle in ("all", "*"):
        return scenarios
    # comma-separated multiple filters → OR
    needles = [n.strip() for n in needle.split(",") if n.strip()]
    if not needles:
        return scenarios
    return [s for s in scenarios if any(n in s.id.lower() for n in needles)]
