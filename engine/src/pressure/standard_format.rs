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

/// 내부 가변 상태 — `KVCache` 와 비-F32 cast scratch 를 **단일 lock** 으로 묶는다.
///
/// scratch(`k_cast`/`v_cast`)는 비-F32 write 경로의 reusable buffer 로, `forward_gen` 의
/// `ws.k_cast`/`ws.v_cast` 와 같은 역할(토큰마다 재할당 방지). cache 와 한 `Mutex` 안에 두어
/// 별도 lock 으로 인한 동시성 hazard 를 원천 차단한다(write 가 cache+scratch 를 항상 함께 만짐).
struct StandardFormatInner {
    cache: KVCache,
    /// Lazy cast scratch (target dtype). 첫 비-F32 write 에서 inner cache 의 allocator 로 할당.
    k_cast: Option<Tensor>,
    v_cast: Option<Tensor>,
}

/// Standard (F32/F16/Q4_0) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `KVCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `KVCache` 자체는 무변.
pub struct StandardFormat {
    idx: usize,
    inner: Mutex<StandardFormatInner>,
}

impl StandardFormat {
    /// `KVCache` 를 layer 인덱스와 함께 wrapping. (현재 unit test 전용 — unwired.)
    pub fn new(idx: usize, inner: KVCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(StandardFormatInner {
                cache: inner,
                k_cast: None,
                v_cast: None,
            }),
        }
    }

    /// KV write 흡수 — `forward_gen` 의 KV-update 분기(transformer_layer/forward_gen.rs:330-386)를
    /// format 표면으로 옮긴 것. `is_decode`(seq_len=1)면 GPU fused cast+scatter fast-path 게이팅.
    ///
    /// **host 경로 = correctness fallback** — `CpuBackend` 는 `is_gpu()==false`라 GPU scatter 분기를
    /// 밟지 않으므로 host build+test 가 F32/비-F32 cast 경로를 검증하고, GPU scatter 정확성은
    /// device round(substep (3c))에서 검증한다. **비-F32(F16/Q4_0) cast 경로**(forward_gen 의
    /// `memory.alloc` + `ws.k_cast` scratch)는 inner `KVCache` 의 allocator 로 scratch 를 lazy 할당해
    /// 흡수한다 — write_kv signature 에 `memory` 를 추가하지 않는다(format⊥hardware, KVCache 가 이미
    /// 동일 allocator 보유). 여전히 unwired 라 무회귀(production 호출처 0).
    fn write_inner(
        &self,
        new_k: &Tensor,
        new_v: &Tensor,
        backend: &dyn Backend,
        is_decode: bool,
    ) -> Result<()> {
        use crate::kv_cache_ops::KVLayout;

        let mut guard = self.inner.lock().unwrap();
        let kv_dtype = guard.cache.kv_dtype();

        // GPU F16 HeadMajor decode: fused cast+scatter (1 dispatch). host 미진입(is_gpu=false).
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F16
            && guard.cache.layout() == KVLayout::HeadMajor
        {
            let cache = &mut guard.cache;
            let pos = KVCacheOps::current_pos(&*cache);
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16(new_k, new_v, k_buf, v_buf, head_dim, cap, pos)?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // GPU F32 HeadMajor decode: single batched scatter dispatch.
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F32
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_f32_batch()
        {
            let cache = &mut guard.cache;
            let pos = KVCacheOps::current_pos(&*cache);
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f32_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, 1,
                )?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // 비-F32 cast 경로: F32 입력을 cache dtype(F16/Q4_0)으로 cast 후 update. (forward_gen 의
        // `kv_dtype != F32` 분기 흡수.) `KVCache::update` 는 cast 를 하지 않고 입력이 이미 cache dtype
        // 임을 전제하므로, scatter fast-path 에 안 잡힌 비-F32 write 는 반드시 여기서 cast 해야 한다
        // (Q4_0 은 GPU 에서도 fast-path 부재라 이 경로). dtype 미일치 silent garbage 방지.
        if kv_dtype != DType::F32 {
            let memory = guard.cache.memory().ok_or_else(|| {
                anyhow::anyhow!(
                    "StandardFormat: non-F32 cast write requires a dynamic KVCache (memory=Some); \
                     fully pre-allocated caches built via KVCache::new() cannot allocate cast scratch"
                )
            })?;
            // scratch lazy 할당 (target dtype). 동일 shape 면 재사용(decode 연속 토큰=seq1 고정),
            // shape 가 바뀌면 재할당한다 — write_kv(decode seq=1)와 write_kv_batch(prefill seq>1)가
            // 같은 format 에서 cast 분기를 공유하므로(K/V 는 KV 불변식상 동일 shape), 첫 write 의
            // 크기로 굳히면 batch↔decode 혼용 시 cast zip 절단·update 오동작. (forward_gen 은
            // decode-only 라 단일 크기였음.)
            let n_elem: usize = new_k.shape().dims().iter().product();
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => {
                    (n_elem / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
                }
                _ => n_elem * 4,
            };
            let k_stale = guard
                .k_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_k.shape().dims());
            if k_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.k_cast = Some(Tensor::new(
                    new_k.shape().clone(),
                    buf,
                    new_k.backend().clone(),
                ));
            }
            let v_stale = guard
                .v_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_v.shape().dims());
            if v_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.v_cast = Some(Tensor::new(
                    new_v.shape().clone(),
                    buf,
                    new_v.backend().clone(),
                ));
            }
            // 필드별 독립 mutable borrow (cache + scratch 동시 접근).
            let StandardFormatInner {
                cache,
                k_cast,
                v_cast,
            } = &mut *guard;
            let k_cast = k_cast.as_mut().unwrap();
            let v_cast = v_cast.as_mut().unwrap();
            backend.cast(new_k, k_cast)?;
            backend.cast(new_v, v_cast)?;
            return cache.update(k_cast, v_cast);
        }

        // Correctness/CPU F32 경로: `KVCache` 는 GPU-buffer 보유라 `update` 가 내부 backend 로 자체 처리
        // (구 `update_kv_cache` 의 has_gpu_buffers 분기).
        guard.cache.update(new_k, new_v)
    }
}

impl KVCacheFormat for StandardFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        KVCacheOps::current_pos(&self.inner.lock().unwrap().cache)
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().cache.capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // decode (seq_len=1) — GPU fused cast+scatter fast-path 게이팅 가능.
        self.write_inner(new_k, new_v, backend, true)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // prefill (seq_len>1) — decode fast-path 미적용(GPU prefill batch scatter 흡수는 후속 substep).
        self.write_inner(new_k, new_v, backend, false)
    }

    fn compact(&self, keep: &[usize], merges: &[Merge]) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();

        // Step 1 (merges): 가중 병합을 compaction 이전 좌표계에서 buffer 에 in-place 적용.
        // F32/F16 만 지원(CPU-accessible buffer 전제). Q4_0 은 dequant+requant 비용으로 merge
        // 스킵(D2O 의 GPU-only merge_enabled=false 와 동일한 보수적 fallback) — keep compaction 만.
        if !merges.is_empty() {
            apply_merges(&mut guard.cache, merges);
        }

        // Step 2 (keep): retained 토큰을 앞으로 당김. write_start=0 으로 전체 재배치
        // (compact 의 keep 은 절대 위치 목록, ascending 가정).
        guard.cache.compact_keep_positions(keep, 0)?;
        guard.cache.set_current_pos(keep.len());
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
        let mut guard = self.inner.lock().unwrap();
        let cache = &mut guard.cache;
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

    fn f16_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        use half::f16;
        let buf = Arc::new(SharedBuffer::new(data.len() * 2, DType::F16));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        for (d, &s) in t.as_mut_slice::<f16>().iter_mut().zip(data.iter()) {
            *d = f16::from_f32(s);
        }
        t
    }

    /// Build a SeqMajor F16 *dynamic* KVCache with a real allocator (`memory=Some`),
    /// so the non-F32 cast scratch path can lazily allocate its scratch buffers.
    fn make_f16_dynamic_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        let total = max_seq * kv_heads * head_dim;
        let k = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let v = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        KVCache::new_dynamic(k, v, max_seq, max_seq, kv_heads, head_dim, mem)
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
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // batch write: 2 tokens
        let batch = vec![2.0f32; 2 * kv_heads * head_dim];
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_write_kv_f16_casts_f32_input() {
        use half::f16;
        // F16 cache + CpuBackend(is_gpu()==false) → 비-F32 cast 분기(GPU scatter fast-path 미진입).
        // F32 입력이 F16 으로 cast 되어 저장되는지 검증 — `KVCache::update` 는 cast 를 안 하므로
        // 이 흡수가 빠지면 dtype 미일치 silent garbage. (forward_gen 의 `kv_dtype != F32` 흡수.)
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // F16 로 정확히 표현 가능한 값(0.5 배수).
        let token0: Vec<f32> = (0..row).map(|i| (i as f32) * 0.5).collect();
        let k0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        let v0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        fmt.write_kv(&k0, &v0, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // 두 번째 토큰 — lazy scratch 재사용 경로(k_cast/v_cast 이미 Some).
        let token1: Vec<f32> = (0..row).map(|i| (i as f32) + 1.0).collect();
        let k1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        let v1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        fmt.write_kv(&k1, &v1, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // F16 buffer 검증: SeqMajor 라 pos*row + idx.
        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        let v16 = guard.cache.v_buffer.as_slice::<f16>();
        for (i, &exp) in token0.iter().enumerate() {
            assert!(
                (k16[i].to_f32() - exp).abs() < 1e-3,
                "pos0 K[{i}] expected {exp}, got {}",
                k16[i].to_f32()
            );
            assert!((v16[i].to_f32() - exp).abs() < 1e-3);
        }
        for (i, &exp) in token1.iter().enumerate() {
            assert!(
                (k16[row + i].to_f32() - exp).abs() < 1e-3,
                "pos1 K[{i}] expected {exp}, got {}",
                k16[row + i].to_f32()
            );
            assert!((v16[row + i].to_f32() - exp).abs() < 1e-3);
        }
    }

    #[test]
    fn test_write_kv_f16_batch_then_decode_reallocs_scratch() {
        use half::f16;
        // write_kv_batch(seq=2) 가 cast scratch 를 seq=2 크기로 굳힌 뒤 write_kv(seq=1) 가 와도
        // scratch 가 shape 변화에 맞춰 재할당되어 둘 다 정확해야 한다(가드 부재 시 cast zip 절단).
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // prefill batch: 2 tokens. token@pos p = 0.5*(p+1) 균일.
        let batch: Vec<f32> = (0..2 * row)
            .map(|i| if i < row { 0.5 } else { 1.0 })
            .collect();
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // decode single: shape [1,1,...] — scratch 재할당 트리거.
        let dec = vec![2.5f32; row];
        let kd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        let vd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        fmt.write_kv(&kd, &vd, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);

        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        // pos0 = 0.5, pos1 = 1.0 (batch), pos2 = 2.5 (decode).
        for i in 0..row {
            assert!(
                (k16[i].to_f32() - 0.5).abs() < 1e-3,
                "pos0[{i}]={}",
                k16[i].to_f32()
            );
            assert!(
                (k16[row + i].to_f32() - 1.0).abs() < 1e-3,
                "pos1[{i}]={}",
                k16[row + i].to_f32()
            );
            assert!(
                (k16[2 * row + i].to_f32() - 2.5).abs() < 1e-3,
                "pos2[{i}]={}",
                k16[2 * row + i].to_f32()
            );
        }
    }

    #[test]
    fn test_write_kv_f16_requires_dynamic_cache() {
        // 비-F32 cast 는 inner cache 의 allocator 가 필요. `KVCache::new()`(memory=None)로 만든
        // F16 cache 는 scratch 할당 불가 → 명시적 에러(silent 오동작 금지).
        let kv_heads = 1;
        let head_dim = 4;
        let total = 8 * kv_heads * head_dim;
        let buf_k = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let buf_v = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let k = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_k,
            Arc::new(CpuBackend::new()),
        );
        let v = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_v,
            Arc::new(CpuBackend::new()),
        );
        let fmt = StandardFormat::new(0, KVCache::new(k, v, 8)); // memory=None

        let token = vec![1.0f32; kv_heads * head_dim];
        let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let err = fmt.write_kv(&kt, &vt, &CpuBackend::new());
        assert!(
            err.is_err(),
            "F16 cast on pre-allocated (memory=None) cache must error"
        );
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
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt.current_pos(), 4);

        // Keep positions 0 and 2 (drop 1, 3); no merges.
        fmt.compact(&[0, 2], &[]).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // Verify buffer layout: pos0 = token0 (unchanged), pos1 = token2 (moved from 2).
        let guard = fmt.inner.lock().unwrap();
        let k = guard.cache.k_buffer.as_slice::<f32>();
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
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }

        // Merge token 3 into token 1, then keep {0, 1, 2}.
        // pos1 (pre-compact) becomes mean(10, 6) = 8.
        let merges = vec![Merge {
            into: 1,
            from: vec![3],
        }];
        fmt.compact(&[0, 1, 2], &merges).unwrap();
        assert_eq!(fmt.current_pos(), 3);

        let guard = fmt.inner.lock().unwrap();
        let k = guard.cache.k_buffer.as_slice::<f32>();
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
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
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
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
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
