//! P2d — Demote 모사 측정 하네스 (KV roadmap 항목 0 §4.4).
//!
//! 동일 메모리 예산에서:
//!   (a) sliding N개 토큰 F16 유지 (나머지 evict)
//!   (b) 2N~4N개 토큰 보존 — sliding 창 밖 후보를 Q4(또는 Q2)로 demote
//!
//! demote 후보를 in-place quant→dequant 왕복하여 정밀도 손실 주입 후 NMSE 비교.
//!
//! ## 설계서 §4.4-D 모사 한계 (명시)
//!
//! quant→dequant 왕복은 품질상 mixed-precision 저장과 수치 등가이나,
//! K 손실의 차기 step 2차 효과(attention weight 교란)는 미포착한다.
//! 본 게이트는 V 손실 + K 직접 손실의 1차 효과만 측정한다.
//!
//! ## 완료 게이트
//!
//! ① 왕복 수치 정합(왕복 후 값 = quantizer dequant 결과와 일치)
//! ② 동일 메모리 예산: (a) vs (b) NMSE 비교 수치 산출
//! ③ 엔진 lib 코드 무수정 확인 (이 파일이 tests/에 격리됨)
//! ④ Q4 vs Q2 격차 측정 (논문 2-bit 급락 재현 여부)

use llm_rs2::qcf::quant_qcf::compute_nmse_block;
use llm_rs2::quant::{BlockKVQ4, BlockQ2_0, QKKV};

// ── 공통 helpers ────────────────────────────────────────────────────────────

/// 단일 Q4 블록 quant→dequant 왕복.
fn demote_q4_block(src: &[f32; QKKV]) -> [f32; QKKV] {
    let block = BlockKVQ4::quantize(src);
    let mut out = [0.0f32; QKKV];
    block.dequantize(&mut out);
    out
}

/// 단일 Q2 블록 quant→dequant 왕복.
fn demote_q2_block(src: &[f32; QKKV]) -> [f32; QKKV] {
    let block = BlockQ2_0::quantize(src);
    let mut out = [0.0f32; QKKV];
    block.dequantize(&mut out);
    out
}

/// K cache per-channel quant→dequant 왕복.
///
/// K layout: `[kv_heads][n_tokens][head_dim]`.
/// per-channel = head_dim 축 기준 — 동일 채널(col) 여러 토큰을 1블록으로.
/// QKKV=32 기준: n_tokens가 32의 배수일 때 각 채널을 32 토큰씩 quantize.
///
/// 반환: 왕복 후 K values (원본과 같은 크기).
fn demote_k_per_channel(
    k: &[f32],
    n_tokens: usize,
    head_dim: usize,
    kv_heads: usize,
    bits: u8,
) -> Vec<f32> {
    let mut out = k.to_vec();
    let n_groups = n_tokens / QKKV;
    if n_groups == 0 {
        return out; // n_tokens < QKKV → skip
    }

    for h in 0..kv_heads {
        let head_base = h * n_tokens * head_dim;
        for ch in 0..head_dim {
            for group in 0..n_groups {
                let tok_start = group * QKKV;
                let mut vals = [0.0f32; QKKV];
                for (t, v) in vals.iter_mut().enumerate() {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    *v = k[idx];
                }
                let reconstructed = match bits {
                    4 => demote_q4_block(&vals),
                    2 => demote_q2_block(&vals),
                    _ => vals,
                };
                for (t, &r) in reconstructed.iter().enumerate() {
                    let idx = head_base + (tok_start + t) * head_dim + ch;
                    out[idx] = r;
                }
            }
        }
    }
    out
}

/// V cache per-token quant→dequant 왕복.
///
/// V layout: `[kv_heads][n_tokens][head_dim]`.
/// per-token = 각 토큰의 head_dim 요소를 head_dim/QKKV 블록으로 quantize.
fn demote_v_per_token(
    v: &[f32],
    n_tokens: usize,
    head_dim: usize,
    kv_heads: usize,
    bits: u8,
) -> Vec<f32> {
    let mut out = v.to_vec();
    let blocks_per_token = head_dim / QKKV;
    if blocks_per_token == 0 {
        return out; // head_dim < QKKV → skip
    }

    for h in 0..kv_heads {
        let head_base = h * n_tokens * head_dim;
        for t in 0..n_tokens {
            let tok_base = head_base + t * head_dim;
            for b in 0..blocks_per_token {
                let start = tok_base + b * QKKV;
                let chunk: &[f32; QKKV] = v[start..start + QKKV].try_into().unwrap();
                let reconstructed = match bits {
                    4 => demote_q4_block(chunk),
                    2 => demote_q2_block(chunk),
                    _ => *chunk,
                };
                out[start..start + QKKV].copy_from_slice(&reconstructed);
            }
        }
    }
    out
}

/// K 또는 V 배열에 대해 전체 NMSE를 계산한다 (블록 평균).
fn mean_nmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n_blocks = original.len() / QKKV;
    if n_blocks == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for b in 0..n_blocks {
        let start = b * QKKV;
        let orig: &[f32; QKKV] = original[start..start + QKKV].try_into().unwrap();
        let recon: &[f32; QKKV] = reconstructed[start..start + QKKV].try_into().unwrap();
        // MSE / Var 방식으로 직접 계산
        let nmse = compute_nmse_block_pair(orig, recon);
        sum += nmse;
    }
    sum / n_blocks as f32
}

/// original vs reconstructed 배열 쌍에 대한 NMSE 계산.
fn compute_nmse_block_pair(original: &[f32; QKKV], reconstructed: &[f32; QKKV]) -> f32 {
    let mean = original.iter().sum::<f32>() / QKKV as f32;
    let var = original.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / QKKV as f32;
    if var < 1e-8 {
        return 0.0;
    }
    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum::<f32>()
        / QKKV as f32;
    (mse / var).clamp(0.0, 1.0)
}

// ── 게이트 ① : 왕복 수치 정합 테스트 ──────────────────────────────────────

/// Q4 왕복: demote_q4_block 결과가 BlockKVQ4::quantize→dequantize와 bit-identical.
#[test]
fn test_demote_q4_roundtrip_matches_quantizer() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.1);
    let roundtrip = demote_q4_block(&src);

    let block = BlockKVQ4::quantize(&src);
    let mut expected = [0.0f32; QKKV];
    block.dequantize(&mut expected);

    for (i, (&a, &b)) in roundtrip.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Q4 roundtrip mismatch at idx {i}: {a} vs {b}"
        );
    }
}

/// Q2 왕복: demote_q2_block 결과가 BlockQ2_0::quantize→dequantize와 bit-identical.
#[test]
fn test_demote_q2_roundtrip_matches_quantizer() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.05 + 0.1);
    let roundtrip = demote_q2_block(&src);

    let block = BlockQ2_0::quantize(&src);
    let mut expected = [0.0f32; QKKV];
    block.dequantize(&mut expected);

    for (i, (&a, &b)) in roundtrip.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Q2 roundtrip mismatch at idx {i}: {a} vs {b}"
        );
    }
}

/// K per-channel 왕복: n_tokens=32(1블록), head_dim=64, kv_heads=1.
#[test]
fn test_demote_k_per_channel_roundtrip() {
    let kv_heads = 1;
    let n_tokens = 32; // 1 QKKV 블록
    let head_dim = 64;
    let n = kv_heads * n_tokens * head_dim;
    let k_orig: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

    let k_q4 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 4);

    // 왕복 후 각 채널의 각 블록은 quantizer 결과와 일치해야 한다.
    for ch in 0..head_dim {
        let mut vals = [0.0f32; QKKV];
        for t in 0..n_tokens {
            vals[t] = k_orig[t * head_dim + ch];
        }
        let expected = demote_q4_block(&vals);
        for t in 0..n_tokens {
            let got = k_q4[t * head_dim + ch];
            assert!(
                (got - expected[t]).abs() < 1e-7,
                "K per-channel mismatch ch={ch} t={t}: {got} vs {}",
                expected[t]
            );
        }
    }
}

/// V per-token 왕복: n_tokens=4, head_dim=64, kv_heads=1.
#[test]
fn test_demote_v_per_token_roundtrip() {
    let kv_heads = 1;
    let n_tokens = 4;
    let head_dim = 64; // 2 블록/토큰
    let n = kv_heads * n_tokens * head_dim;
    let v_orig: Vec<f32> = (0..n).map(|i| ((i % 50) as f32) * 0.1 + 0.5).collect();

    let v_q4 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 4);

    // 왕복 후 각 토큰의 각 블록은 quantizer 결과와 일치해야 한다.
    let blocks_per_token = head_dim / QKKV;
    for t in 0..n_tokens {
        for b in 0..blocks_per_token {
            let start = t * head_dim + b * QKKV;
            let chunk: &[f32; QKKV] = v_orig[start..start + QKKV].try_into().unwrap();
            let expected = demote_q4_block(chunk);
            let got = &v_q4[start..start + QKKV];
            for (j, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
                assert!(
                    (e - g).abs() < 1e-7,
                    "V per-token mismatch t={t} b={b} j={j}: expected={e} got={g}"
                );
            }
        }
    }
}

// ── 게이트 ② : 동일 메모리 예산 (a) vs (b) 비교 ───────────────────────────

/// Demote 스모크 비교: 동일 메모리 예산에서
///   (a) sliding N개 F16 (N개만 보존) vs (b) 4N개 Q4(demote, 4× 토큰)
///
/// NMSE를 비교해 Q4 demote의 정보 보존량을 측정한다.
/// 이 테스트는 demote가 sliding보다 NMSE를 줄이는지(더 많은 토큰 보존)를 확인한다.
#[test]
fn test_demote_vs_sliding_smoke() {
    // 설정: kv_heads=2, n_tokens_budget=32(F16), head_dim=64
    let kv_heads = 2;
    let head_dim = 64;
    let n_tokens_budget = 32; // sliding에서 보존되는 토큰 수 (메모리 기준)

    // Q4는 F16의 약 4× 토큰 저장 가능 (16bit/elem → ~4.5bit/elem)
    // 정수 배수로 간소화: 4×
    let n_tokens_demote = n_tokens_budget * 4; // 128 토큰 Q4 저장

    // 전체 KV cache (n_tokens_demote 토큰)
    let n_total = kv_heads * n_tokens_demote * head_dim;
    let k_full: Vec<f32> = (0..n_total)
        .map(|i| ((i % 200) as f32) * 0.05 - 5.0)
        .collect();
    let v_full: Vec<f32> = (0..n_total)
        .map(|i| ((i % 150) as f32) * 0.03 + 1.0)
        .collect();

    // (a) sliding: 마지막 n_tokens_budget 토큰만 F16 보존 (앞 토큰 버림)
    // 보존 구간: [n_tokens_demote - n_tokens_budget .. n_tokens_demote)
    let sliding_start = n_tokens_demote - n_tokens_budget;
    let mut k_sliding = vec![0.0f32; n_total];
    let mut v_sliding = vec![0.0f32; n_total];
    for h in 0..kv_heads {
        for t in sliding_start..n_tokens_demote {
            let src_base = h * n_tokens_demote * head_dim + t * head_dim;
            let dst_base = h * n_tokens_demote * head_dim + t * head_dim;
            k_sliding[dst_base..dst_base + head_dim]
                .copy_from_slice(&k_full[src_base..src_base + head_dim]);
            v_sliding[dst_base..dst_base + head_dim]
                .copy_from_slice(&v_full[src_base..src_base + head_dim]);
        }
    }

    // (b) demote: 전체 n_tokens_demote 토큰을 Q4로 demote (메모리 = (a)와 동일)
    let k_demoted = demote_k_per_channel(&k_full, n_tokens_demote, head_dim, kv_heads, 4);
    let v_demoted = demote_v_per_token(&v_full, n_tokens_demote, head_dim, kv_heads, 4);

    // NMSE 계산: 보존 구간 기준
    // sliding: 앞 토큰이 버려져 있으므로 전체 토큰 대비 손실 = 버린 토큰의 원본 정보 소실.
    //          보존 구간의 NMSE = 0 (F16 = 무손실), 버린 구간 = 완전 손실(전체 원본 대비).
    // demote: 전체 토큰의 Q4 근사 NMSE.
    let k_demoted_nmse = mean_nmse(&k_full, &k_demoted);
    let v_demoted_nmse = mean_nmse(&v_full, &v_demoted);

    // sliding의 "손실": 버린 토큰을 0으로 채웠으므로 원본 대비 NMSE가 높다.
    let k_sliding_nmse = mean_nmse(&k_full, &k_sliding);
    let v_sliding_nmse = mean_nmse(&v_full, &v_sliding);

    eprintln!(
        "[Demote smoke] K: demote_nmse={:.4} sliding_nmse={:.4}",
        k_demoted_nmse, k_sliding_nmse
    );
    eprintln!(
        "[Demote smoke] V: demote_nmse={:.4} sliding_nmse={:.4}",
        v_demoted_nmse, v_sliding_nmse
    );

    // demote는 전체 토큰을 근사하여 유지하므로 sliding(버린 토큰 = 완전 손실)보다
    // 전체 NMSE가 낮아야 한다.
    assert!(
        k_demoted_nmse < k_sliding_nmse,
        "K: demote ({:.4}) should have lower NMSE than sliding ({:.4}) — \
         demote retains 4× tokens at Q4 precision vs sliding discarding 75%",
        k_demoted_nmse,
        k_sliding_nmse
    );
    assert!(
        v_demoted_nmse < v_sliding_nmse,
        "V: demote ({:.4}) should have lower NMSE than sliding ({:.4})",
        v_demoted_nmse,
        v_sliding_nmse
    );
}

// ── 게이트 ②(보조): Q4 vs Q2 격차 ─────────────────────────────────────────

/// Q4 vs Q2 NMSE 격차: Q2가 Q4보다 오차가 크다 (논문 2-bit 1B 급락 재현 가능성).
#[test]
fn test_q4_vs_q2_nmse_gap() {
    let kv_heads = 1;
    let n_tokens = 64;
    let head_dim = 64;
    let n = kv_heads * n_tokens * head_dim;

    let k_orig: Vec<f32> = (0..n).map(|i| ((i % 100) as f32) * 0.1).collect();
    let v_orig: Vec<f32> = (0..n).map(|i| ((i % 80) as f32) * 0.05 + 0.5).collect();

    let k_q4 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 4);
    let k_q2 = demote_k_per_channel(&k_orig, n_tokens, head_dim, kv_heads, 2);
    let v_q4 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 4);
    let v_q2 = demote_v_per_token(&v_orig, n_tokens, head_dim, kv_heads, 2);

    let k_nmse_q4 = mean_nmse(&k_orig, &k_q4);
    let k_nmse_q2 = mean_nmse(&k_orig, &k_q2);
    let v_nmse_q4 = mean_nmse(&v_orig, &v_q4);
    let v_nmse_q2 = mean_nmse(&v_orig, &v_q2);

    eprintln!(
        "[Q4vsQ2] K: q4={:.4} q2={:.4} | V: q4={:.4} q2={:.4}",
        k_nmse_q4, k_nmse_q2, v_nmse_q4, v_nmse_q2
    );

    assert!(
        k_nmse_q2 >= k_nmse_q4,
        "K: Q2 NMSE ({:.4}) should be >= Q4 NMSE ({:.4})",
        k_nmse_q2,
        k_nmse_q4
    );
    assert!(
        v_nmse_q2 >= v_nmse_q4,
        "V: Q2 NMSE ({:.4}) should be >= Q4 NMSE ({:.4})",
        v_nmse_q2,
        v_nmse_q4
    );
}

/// compute_nmse_block 재사용 정합: demote 모사의 단일 블록 NMSE가
/// quant_qcf.rs::compute_nmse_block과 일치한다.
#[test]
fn test_nmse_consistency_with_quant_qcf() {
    let src: [f32; QKKV] = std::array::from_fn(|i| (i as f32) * 0.1);

    // compute_nmse_block (quant_qcf.rs 정의)
    let nmse_from_lib = compute_nmse_block(&src, 4, 1e-8);

    // 직접 계산 (demote_measure 내부 compute_nmse_block_pair 이용)
    let roundtrip = demote_q4_block(&src);
    let nmse_direct = compute_nmse_block_pair(&src, &roundtrip);

    assert!(
        (nmse_from_lib - nmse_direct).abs() < 1e-5,
        "NMSE consistency: lib={:.6} direct={:.6}",
        nmse_from_lib,
        nmse_direct
    );
}
