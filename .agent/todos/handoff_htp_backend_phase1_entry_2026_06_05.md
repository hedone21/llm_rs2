# Handoff: HTP backend S4+S5+Y — S25 device 검증 완료 (전 게이트 GREEN)

**작성**: 2026-06-05 (갱신: /loop dynamic — device 검증 D→1→2 전부 GREEN)
**HEAD**: `5b5fe752` 직후 본 커밋 (device 검증 결과 반영)
**HTP 커밋**: `d331d01b`(S4+S5) + `eab1d65e`(Y) + `5b5fe752`(handoff) — origin/master push 완료
**브랜치**: `master`
**작성자**: 메인 세션 (오케스트레이터)

**다음 세션 진입 문장**: "HTP backend device-GREEN 완료 — 3B OOM 헤드룸 / prof_usecs 노출 / paper evidence 중 택1"

---

## TL;DR

HTP FastRPC backend 의 **prefill+decode weight matmul NPU dispatch(S4+S5+Y)가 S25 device 에서 전 게이트 GREEN**. Qwen2.5-1.5B Q4_0 로 D(microbench dispatch 격리)→1(OOM+부팅+M>1)→2(token-id 정합) 3단계 검증 완료. 핵심: **OOM 안 남**(~703MB rpcmem q4x4x2 상주 성공), **M>1 GEMM DSP 작동**(prefill 5tok), **token-id 16/16 CPU 와 완전 일치**. 이번 라운드 코드 변경 0 — env 수정만으로 GREEN. paper angle = quality-aware heterogeneous placement (NPU decode 20.9 < CPU 30.0 tok/s, 느려도 정합).

**환경 발견(중요)**: `--backend htp` 는 `ADSP_LIBRARY_PATH=<work_dir>` 필수 (DSP skel `libggml-htp-v79.so` 탐색). run_device.py 는 미설정 → 수동 adb 또는 `microbench_qnn_matrix.py` 드라이버 env 사용.

---

## 진행 상태 — device 검증 완료 (2026-06-05)

| 게이트 | 작업 | 상태 | 증거 |
|---|---|---|---|
| D | microbench dispatch 격리 | 🟢 GREEN | prof_usecs=159, prof_cycles=336430, max_abs_err=2.22e-2(<5e-2), mm_qkv K=1536 N=2048 |
| 1 | OOM + 부팅 + M>1 | 🟢 GREEN | `NPU dispatch 활성` 출력, OOM 0, prefill 5tok(M>1) 작동, 출력 coherent |
| 2 | token-id 정합 | 🟢 GREEN | HTP=CPU **16/16 토큰 일치** (first=12095), "Paris. It has a population of about 2 million..." |

**측정 (Qwen2.5-1.5B Q4_0, S25 6T, taskset 3f)**:
- HTP : TTFT 105 ms, Decode **47.81 ms/tok (20.9 tok/s)**, AvgTBT 51.4 ms
- CPU : TTFT 617 ms, Decode **33.33 ms/tok (30.0 tok/s)**, AvgTBT 69.8 ms
- NPU decode 가 CPU 대비 ~1.4× 느림 (예측 NPU<GPU<CPU 일치). cpu fallback 1건 = lm_head(token_embd 파생 Q4_0, 비-rpcmem → standard 안전 읽기, garbage 아님).

**코드 상태**: S1~S5+Y 전부 master push 완료. host 게이트(fmt/clippy/lib-test 8/8/NDK) + device 게이트(D/1/2) 모두 PASS. **이번 검증에서 코드 수정 0** (env 만으로 GREEN).

---

## 다음 작업 (선택, phase1 완료 후)

phase 1(HTP bring-up S1~S5+Y)은 **device-GREEN 으로 완료**. 다음은 우선순위 미정 — 택1:

1. **3B OOM 헤드룸**: `llama3.2-3b-q4_0.gguf`(이미 device 상주, 큰 ffn) 로 rpcmem 상한 탐색. RED 시 O4(ffn host-heap hybrid, ~6줄).
2. **prof_usecs 노출**: `via_htp` 가 `rsp.prof_usecs` 버림 → 누적 노출(~3줄)로 production TBT 에 NPU 시간 가시화.
3. **paper evidence**: M3/M4/M5 microbench 와 통합, heterogeneous placement quality table 작성.

---

## Landmines / 해소 기록

- **✅ OOM (해소)**: Qwen2.5-1.5B Q4_0 ~703MB rpcmem q4x4x2 상주 성공. NULL/mmap-fail 0. **단 3B+ 미검증** → 다음 작업 1.
- **✅ M>1 GEMM DSP (해소)**: prefill 5tok(M=5)에서 NPU dispatch + 정상 출력. DSP general matmul 지원 확인 (loud bail / status 에러 0).
- **✅ token-id 정합 (해소)**: 16/16 일치. "logit max_abs_err<1e-2" 게이트 폐기 정당 (token-id 가 강한 신호).
- **★ ADSP_LIBRARY_PATH 필수 (신규 — 첫 실행 RED 원인)**: `--backend htp`/microbench 실행 시 `ADSP_LIBRARY_PATH=/data/local/tmp` 없으면 `remote_handle64_open` 실패(SKIP). run_device.py 미설정 → 수동 adb shell 사용:
  ```bash
  adb -s R3CY408S5SB shell "cd /data/local/tmp && \
    LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
    ADSP_LIBRARY_PATH=/data/local/tmp taskset 3f \
    ./argus_cli --backend htp \
      --model-path models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-q4_0.gguf \
      --tokenizer-path models/qwen2.5-1.5b-gguf/tokenizer.json \
      --prompt 'The capital of France is' --num-tokens 16 --greedy"
  ```
  (구버전 handoff 의 `LD_LIBRARY_PATH=.:/vendor/lib64` 만으로는 부족.) 빌드+배포는 `python scripts/run_device.py -d galaxy_s25 --skip-exec argus_cli` (features 는 devices.toml 자동 주입).
- **1-token 프롬프트 함정**: 검증은 멀티토큰('The capital of France is' 5tok)으로 수행 — prefill garbage 가려짐 회피 확인.
- **q4x4x2 ↔ standard 비호환**: Y override 가 q4x4x2 weight 의 cpu fallback 을 loud bail 로 봉쇄. 비-rpcmem weight(lm_head)는 standard 안전 읽기 — 검증서 cpu fallback 1건 정상.
- **weight-swap 비호환(scope 밖)**: 현 검증은 resilience swap off. swap + q4x4x2 동시 사용 시 미검증.
- **prof_usecs 미노출**: `via_htp` 는 버림. microbench 만 prof 출력(159us). 다음 작업 2.
- **NPU 절대속도 < CPU** (20.9 < 30.0 tok/s). "빠르다" 금지 — 협력+품질 서사.

---

## 참고 링크

- 메모리: `[[htp-device-verification-green]]`(본 결과+env 레시피), `[[htp-production-backend-feasibility]]`, `[[htp-f16-matmul-measurement-artifact]]`
- device 실행 env 레시피: `scripts/microbench_qnn_matrix.py:186` (ADSP_LIBRARY_PATH 주입) / 본 handoff Landmines
- 코드: `engine/src/backend/htp_fastrpc.rs` (matmul_transposed/via_htp/dispatchable, copy_weight_from), `.../repack.rs` (q4x4x2), `.../host.rs` (FastRPC 세션)
- GREEN reference: `engine/microbench/htp_matmul.rs::run_htp` (M==1 GEMV)
- 설계: `arch/htp_fastrpc.md`, `spec/htp_fastrpc.md` (INV-HTP-FRPC-001~005)
