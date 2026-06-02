//! `StandardFormat` — `KVCacheFormat` impl wrapping a standard `KVCache` (§4.1, Phase α-K).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 / §2.1 (guard rail: format impl 은 `kv/`
//! (현 `pressure/`)에, base trait 은 `format/` 에).
//!
//! **purely additive, host-only, unwired** — 기존 `KVCache` 와 `KVCacheOps` 경로를 1바이트도
//! 건드리지 않고, 신규 wrapper 로 공존한다. production 에서 `StandardFormat` 를 생성하는 코드는
//! 0(unit test 에서만 생성). 내부 가변성 = `std::sync::Mutex`(trait `Send+Sync` 요구로 `RefCell`
//! 불가; §4.1 R4 상 cold-path 라 lock 비용 무관).

use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::format::{AttnDims, KVCacheFormat, Merge};
use crate::kv_cache_ops::KVCacheOps;
use crate::pressure::kv_cache::KVCache;
use crate::tensor::Tensor;

/// Standard (F32/F16/Q4_0) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `KVCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `KVCache` 자체는 무변.
pub struct StandardFormat {
    idx: usize,
    inner: Mutex<KVCache>,
}

impl StandardFormat {
    /// `KVCache` 를 layer 인덱스와 함께 wrapping. (현재 unit test 전용 — unwired.)
    pub fn new(idx: usize, inner: KVCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(inner),
        }
    }
}

impl KVCacheFormat for StandardFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        KVCacheOps::current_pos(&*self.inner.lock().unwrap())
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        // 단일 토큰 write — geometry 상 batch 와 동일 경로(KVCache::update 가 seq_len 으로 분기).
        self.inner.lock().unwrap().update(new_k, new_v)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        self.inner.lock().unwrap().update(new_k, new_v)
    }

    fn compact(&self, keep: &[usize], merges: &[Merge]) -> Result<()> {
        let mut cache = self.inner.lock().unwrap();

        // Step 1 (merges): 가중 병합을 compaction 이전 좌표계에서 buffer 에 in-place 적용.
        // F32/F16 만 지원(CPU-accessible buffer 전제). Q4_0 은 dequant+requant 비용으로 merge
        // 스킵(D2O 의 GPU-only merge_enabled=false 와 동일한 보수적 fallback) — keep compaction 만.
        if !merges.is_empty() {
            apply_merges(&mut cache, merges);
        }

        // Step 2 (keep): retained 토큰을 앞으로 당김. write_start=0 으로 전체 재배치
        // (compact 의 keep 은 절대 위치 목록, ascending 가정).
        cache.compact_keep_positions(keep, 0)?;
        cache.set_current_pos(keep.len());
        Ok(())
    }

    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        let mut cache = self.inner.lock().unwrap();
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let cache_seq_len = KVCacheOps::current_pos(&*cache);

        // Sliding window: 최근 window 토큰으로 제한 (Gemma3 local). global 이면 전체.
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };

        let (k_cache, v_cache) = KVCacheOps::get_view(&mut *cache);

        // typed/F32 경로: backend.attention_gen 에 위임. CPU backend 는 F32/F16/Q4_0 을 dtype-aware
        // 하게 처리(default impl=F32/F16, CpuBackend override 가 Q4_0 등 흡수). GPU backend 는
        // 자기 커널로 dispatch(host 미검증 — device 검증은 후속 wiring substep).
        //
        // NOTE: kivi-native(get_kivi_raw_buffers) 분기는 KIVIFormat 소관이라 여기 없음. Q4_0+GPU
        // CPU-dequant fallback 의 정밀 재현(attention_q4_gpu_fallback)도 wiring substep 으로 연기 —
        // 본 substep 은 CPU-testable F32/F16 경로의 정확성만 담보한다.
        backend.attention_gen(
            q,
            &k_cache,
            &v_cache,
            out,
            dims.n_heads_q,
            n_heads_kv,
            head_dim,
            effective_cache_len,
            scores,
        )
    }
}

/// `Merge` 목록을 KVCache buffer 에 in-place 적용 (F32/F16 한정).
///
/// 의미: `into` 토큰 = `into` + Σ `from` 의 균등 평균(group 정규화). D2O 의 Eq.11 가중(EMA·sim
/// 기반)은 `D2OHandler` config 책임(§4.1 `Merge` 주석)이라, 본 base impl 은 가중치 미지정 시의
/// 중립 기본(균등)으로 병합한다. compaction 이전 좌표계에서 `cache.offset(pos, head)` 로 원위치를
/// 읽는다(`scatter_reduce_merge_layer_wide` 와 동일 좌표 계약).
fn apply_merges(cache: &mut KVCache, merges: &[Merge]) {
    let dtype = cache.k_buffer.dtype();
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();

    match dtype {
        DType::F32 => {
            for m in merges {
                if m.from.is_empty() {
                    continue;
                }
                let n = (1 + m.from.len()) as f32;
                let w = 1.0 / n;
                for h in 0..kv_heads {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        m.from.iter().map(|&p| cache.offset(p, h)).collect();
                    {
                        let k = cache.k_buffer.as_mut_slice::<f32>();
                        merge_row_f32(k, into_off, &from_offs, head_dim, w);
                    }
                    {
                        let v = cache.v_buffer.as_mut_slice::<f32>();
                        merge_row_f32(v, into_off, &from_offs, head_dim, w);
                    }
                }
            }
        }
        DType::F16 => {
            use half::f16;
            for m in merges {
                if m.from.is_empty() {
                    continue;
                }
                let n = (1 + m.from.len()) as f32;
                let w = 1.0 / n;
                for h in 0..kv_heads {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        m.from.iter().map(|&p| cache.offset(p, h)).collect();
                    {
                        let k = cache.k_buffer.as_mut_slice::<f16>();
                        merge_row_f16(k, into_off, &from_offs, head_dim, w);
                    }
                    {
                        let v = cache.v_buffer.as_mut_slice::<f16>();
                        merge_row_f16(v, into_off, &from_offs, head_dim, w);
                    }
                }
            }
        }
        // Q4_0 등: merge 스킵(eviction-only fallback). keep compaction 만 적용된다.
        _ => {}
    }
}

#[inline]
fn merge_row_f32(buf: &mut [f32], into_off: usize, from_offs: &[usize], head_dim: usize, w: f32) {
    for d in 0..head_dim {
        let mut acc = w * buf[into_off + d];
        for &fo in from_offs {
            acc += w * buf[fo + d];
        }
        buf[into_off + d] = acc;
    }
}

#[inline]
fn merge_row_f16(
    buf: &mut [half::f16],
    into_off: usize,
    from_offs: &[usize],
    head_dim: usize,
    w: f32,
) {
    use half::f16;
    for d in 0..head_dim {
        let mut acc = w * buf[into_off + d].to_f32();
        for &fo in from_offs {
            acc += w * buf[fo + d].to_f32();
        }
        buf[into_off + d] = f16::from_f32(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    /// Build a SeqMajor F32 KVCache: [1, max_seq, kv_heads, head_dim].
    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let total = max_seq * kv_heads * head_dim;
        let k = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        let v = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        KVCache::new(k, v, max_seq)
    }

    #[test]
    fn test_geometry_delegates_to_kvcache() {
        let cache = make_cache(8, 2, 4);
        let fmt = StandardFormat::new(3, cache);
        assert_eq!(fmt.idx(), 3);
        assert_eq!(fmt.capacity(), 8);
        assert_eq!(fmt.current_pos(), 0);
    }

    #[test]
    fn test_write_kv_advances_pos() {
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // single-token write: [1, 1, kv_heads, head_dim]
        let token = vec![1.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        fmt.write_kv(&k, &v).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // batch write: 2 tokens
        let batch = vec![2.0f32; 2 * kv_heads * head_dim];
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_compact_keep_only() {
        let kv_heads = 1;
        let head_dim = 2;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // Write 4 distinct tokens: token p has value [p, p].
        for p in 0..4 {
            let t = vec![p as f32; kv_heads * head_dim];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            fmt.write_kv(&k, &v).unwrap();
        }
        assert_eq!(fmt.current_pos(), 4);

        // Keep positions 0 and 2 (drop 1, 3); no merges.
        fmt.compact(&[0, 2], &[]).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // Verify buffer layout: pos0 = token0 (unchanged), pos1 = token2 (moved from 2).
        let cache = fmt.inner.lock().unwrap();
        let k = cache.k_buffer.as_slice::<f32>();
        assert_eq!(k[0], 0.0, "kept pos0 = token0");
        assert_eq!(k[1], 0.0);
        assert_eq!(k[2], 2.0, "compacted pos1 = token2");
        assert_eq!(k[3], 2.0);
    }

    #[test]
    fn test_compact_with_merge_f32() {
        let kv_heads = 1;
        let head_dim = 2;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // tokens: 0=[0,0] 1=[10,10] 2=[2,2] 3=[6,6]
        let vals = [0.0f32, 10.0, 2.0, 6.0];
        for &p in &vals {
            let t = vec![p; kv_heads * head_dim];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            fmt.write_kv(&k, &v).unwrap();
        }

        // Merge token 3 into token 1, then keep {0, 1, 2}.
        // pos1 (pre-compact) becomes mean(10, 6) = 8.
        let merges = vec![Merge {
            into: 1,
            from: vec![3],
        }];
        fmt.compact(&[0, 1, 2], &merges).unwrap();
        assert_eq!(fmt.current_pos(), 3);

        let cache = fmt.inner.lock().unwrap();
        let k = cache.k_buffer.as_slice::<f32>();
        // pos0=token0=0, pos1=merged(10,6)=8, pos2=token2=2
        assert_eq!(k[0], 0.0);
        assert_eq!(k[2], 8.0, "merged into-token = mean(10,6)");
        assert_eq!(k[4], 2.0);
    }

    #[test]
    fn test_attention_into_f32_uniform() {
        // current_pos==0 is illegal for softmax; write 2 identical tokens so
        // softmax is uniform and output = the (identical) V row.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        let k_row = vec![0.0f32; head_dim]; // zero K → all scores equal → uniform softmax
        let v_row = vec![5.0f32; head_dim];
        for _ in 0..2 {
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_row);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_row);
            fmt.write_kv(&k, &v).unwrap();
        }
        assert_eq!(fmt.current_pos(), 2);

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 2];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            Some(&mut scores),
        )
        .unwrap();

        // Uniform attention over identical V rows → out == V row.
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!((x - 5.0).abs() < 1e-4, "expected 5.0, got {x}");
        }
        // post-softmax scores: 2 equal weights summing to 1.
        assert!((scores[0] - 0.5).abs() < 1e-4);
        assert!((scores[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_attention_into_window_clamps_len() {
        // window=1 must restrict effective_cache_len to 1 (only first token seen
        // by backend.attention_gen). Verify scores buffer reflects single token.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        for p in 0..3 {
            let t = vec![p as f32; head_dim];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            fmt.write_kv(&k, &v).unwrap();
        }

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 3];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: Some(1),
            },
            Some(&mut scores),
        )
        .unwrap();

        // window=1 → only 1 token attended → score[0]=1.0, output = token0 (zeros).
        assert!((scores[0] - 1.0).abs() < 1e-4);
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.abs() < 1e-4, "token0 is all zeros, got {x}");
        }
    }
}
