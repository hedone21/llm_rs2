# cuda_pc SyncPolicy 포팅 — 결론

**날짜**: 2026-04-19
**범위**: cuda_embedded의 SyncCat/SyncPolicy 구조를 cuda_pc로 포팅하여 RTX 3090 Ti 호스트 decode tok/s 개선 검증
**결과**: **성공** — baseline 128.4 tok/s → 142.7 tok/s (**+11.1%**), 6런 모두 정확성 OK

## 핵심 결론

1. **포팅 자체는 재사용 가능**: SyncCat 10-way 분류가 cuda_pc에 그대로 적용됨. 32개 dispatch-path sync 호출이 모두 적절한 카테고리에 들어감 (분류표: `sync_categories.md`).

2. **Jetson UMA의 `minimal` 프리셋은 호스트 discrete GPU에서 불안정**. `elem_add+fallback`만 남기면 런에 따라 정상/garbage가 섞여 나옴. 호스트 managed memory 경로가 UMA와 유사한 coherency 취약성을 보이는 것으로 추정.

3. **호스트 전용 correctness-preserving 최소 세트는 `{ElemAdd, Gather, FallbackPre}`**. Jetson 대비 **`Gather` 카테고리가 추가 필요**. 임베딩 lookup이 동일 스트림이어도 후속 kernel들이 sync 없이는 stale read를 일으킨다. 정확한 원인은 추가 조사 대상이지만, 경험적으로는 재현 가능한 요구 조건.

4. **성능 개선 규모**: 9개 카테고리 중 7개 드롭 → 레이어당 GPU sync 5회 제거 (rms×2 + rope + matmul×5 + attention + kv_scatter + silu_mul ≈ 감소). 호스트 강력한 디코딩 성능(cuBLAS/3090 Ti)에서도 per-op sync 제거 효과가 뚜렷함.

## 권장 사용법

```bash
./target/release/generate ... -b cuda \
    --cuda-sync-policy "custom:elem_add,gather,fallback"
```

또는 CLI에서 간단히 `minimal` 프리셋이 정확성 보장되도록 바꾸는 것이 이상적이나, **cuda_embedded와 CLI 문자열 포맷을 유지하기 위해 프리셋은 변경하지 않음**. 프리셋 호환성 정책: `minimal`은 "Jetson UMA 검증 최소" 의미를 그대로 유지. 호스트에서는 `custom:...`으로 명시.

## 추후 작업 후보

1. **프리셋 `host-minimal` 추가**: `elem_add,gather,fallback` alias. CLI 가독성 개선.
2. **Gather sync 필요성 정밀 분석**: cuBLAS 내부 auxiliary stream 사용 가능성 조사. 필요하면 cuBLAS workspace API로 명시적 스트림 바인딩.
3. **Q4_0 decode 재측정**: 본 벤치는 F16. Q4_0는 `Matmul` → CPU fallback을 타므로 `FallbackPre` sync 빈도가 다름.
4. **NIAH 등 정확성 스트레스 테스트**: 30-token greedy decode는 짧은 시퀀스. 장문 / 수치 민감 태스크에서 correctness 재확인.

## 파일 목록

- `sync_categories.md`: cuda_pc 32개 sync 호출의 SyncCat 분류표
- `results.md`: policy × run 매트릭스, correctness/throughput 데이터
- `policy_all_run[1-3].log`: baseline (all)
- `policy_minimal_run[1-3].log`: Jetson 최적 프리셋 (호스트에서 실패)
- `policy_llamacpp_run[1-3].log`: llamacpp-style (호스트에서 실패)
- `policy_cand9_run[1-6].log`: **최종 권장** `elem_add,gather,fallback` 6런
- `policy_cand11_run[1-6].log`: `elem_add,gather` (fallback 제외, 약간 더 빠르지만 예외 경로 미보장)
- `policy_cand{2-10}_run*.log`, `policy_drop_*.log`, `policy_elemadd_only_run*.log`, `policy_matmul_fb_run*.log`: 탐색 과정 로그

## 커밋

`perf(cuda_pc): port sync-policy from cuda_embedded (+11.1% decode on RTX 3090 Ti)`
