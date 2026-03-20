# 28. Manager Monitor Pattern Redesign

> **⚠️ Superseded**: 이 문서는 `36_policy_design.md`로 대체되었습니다. Monitor 패턴 자체는 채택되었으나, Monitor 이후의 정책 결정 구조가 전면 재설계되었습니다.

## Overview

Manager 서비스를 3-layer (Collector → PolicyEngine → Emitter) 에서 **Monitor 패턴**으로 재설계한다.
각 Monitor가 데이터 수집 + 임계값 평가 + 시그널 생성을 자기완결적으로 수행하며,
Manager 본체는 시그널 버스 역할만 담당한다.

## Motivation

기존 구조에서 `ReadingData` enum이 closed되어 새 지표(FPS, 앱 상태 등) 추가 시
enum 수정 + PolicyEngine 수정이 필요했다 (OCP 위반).
Monitor 패턴에서는 새 Monitor 파일 하나만 추가하면 된다.

## Architecture

```
Before:
  Collector ──Reading──→ PolicyEngine ──SystemSignal──→ Emitter
  (closed ReadingData)   (중앙 집중)

After:
  MemoryMonitor  ──SystemSignal──→ ┐
  ThermalMonitor ──SystemSignal──→ ├→ Emitter
  ComputeMonitor ──SystemSignal──→ ┤  (단순 전달)
  EnergyMonitor  ──SystemSignal──→ ┤
  ExternalMonitor ──SystemSignal──→ ┘
```

## Key Components

### Monitor trait (`monitor/mod.rs`)
- `fn run(&mut self, tx: Sender<SystemSignal>, shutdown: Arc<AtomicBool>)`
- `fn initial_signal(&self) -> Option<SystemSignal>`
- `fn name(&self) -> &str`

### ThresholdEvaluator (`evaluator.rs`)
- 히스테리시스 기반 레벨 평가 유틸리티
- Direction: Ascending (높을수록 나쁨) / Descending (낮을수록 나쁨)
- 각 Monitor가 선택적으로 사용

### ExternalMonitor (`monitor/external.rs`)
- 연구용 시그널 주입점
- Unix socket 또는 stdin에서 JSON 시그널을 읽어 패스스루

## Removed

- `collector/` module (Collector trait, Reading, ReadingData, ResourceKind)
- `policy/` module (PolicyEngine trait, ThresholdPolicy)

## File Structure

```
manager/src/
├── lib.rs
├── main.rs           (signal bus)
├── config.rs         (monitor 중심 설정)
├── evaluator.rs      (ThresholdEvaluator)
├── monitor/
│   ├── mod.rs        (Monitor trait)
│   ├── memory.rs
│   ├── thermal.rs
│   ├── compute.rs
│   ├── energy.rs
│   └── external.rs
└── emitter/          (변경 없음)
```
