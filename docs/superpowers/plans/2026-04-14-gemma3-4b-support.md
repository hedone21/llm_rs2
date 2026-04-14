# Gemma 3 4B 지원 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gemma 3 4B (`google/gemma-3-4b-pt`) multimodal 모델의 text 파트를 llm_rs2에 정식 지원 — config `text_config` auto-flatten + safetensors `language_model.` prefix 자동 strip + 2번째 eval 문항 크래시 근본 원인 해결.

**Architecture:**
- `ModelConfig`에 `weight_prefix: String` 필드를 추가하고, `from_json`이 multimodal wrapper(`Gemma3ForConditionalGeneration` / `model_type=="gemma3"`)를 감지하면 `text_config`를 top-level로 flatten하면서 prefix를 `"language_model."`로 설정한다.
- `WeightMapper` factory (`create_mapper`)는 `ModelConfig`를 받아 Gemma3Mapper 인스턴스에 prefix를 주입한다. 모든 tensor name 생성에 prefix가 붙으므로 safetensors lookup이 정확히 매칭된다.
- `SafetensorsSource::load_raw`는 tensor 미존재 시 상세한 진단 정보(어떤 TensorId→어떤 name으로 매핑되었는지)를 포함한 에러를 반환해, 향후 prefix 이슈 조기 감지.
- 4B-only 크래시(`free(): invalid size`)는 위 경로가 정상화된 뒤에도 재현되는지 먼저 확인하고, 재현되면 AddressSanitizer 빌드로 원인을 국소화한다.

**Tech Stack:** Rust (engine 크레이트), safetensors 0.x, serde_json, Cargo workspace, cargo test, OpenCL 백엔드 (Android 실험).

---

## 참고 컨텍스트 (엔지니어용 사전 학습)

### 이슈 원문
`/home/go/Workspace/papers/pact2026/experiments/issues/gemma3_4b_support_20260413.md` — PACT 2026 QCF 검증용 4번째 모델. 마감 4/24.

### 현재 동작
- **Gemma 3 1B** (`models/gemma3-1b/`): ✅ 정상. config: `Gemma3ForCausalLM` + `model_type: "gemma3_text"` (flat), safetensors: `model.layers.*` prefix.
- **Gemma 3 4B** (`models/gemma3-4b/`): 수동 workaround 적용됨 — `config.json` 이미 flatten (`config.json.orig` 보존), safetensors는 prefix strip + vision 제거된 단일 파일. 그럼에도 eval-ll 2번째 문항 진입 시 `free(): invalid size` 크래시.

### 4B 모델 파라미터
- hidden_size=2560, layers=34, heads=8, kv_heads=4, intermediate=10240, head_dim=256
- `config.json`의 원본 형태:
  ```json
  {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "model_type": "gemma3",
    "text_config": {
      "hidden_size": 2560,
      "num_hidden_layers": 34,
      "model_type": "gemma3_text",
      ...
    },
    "vision_config": { ... }
  }
  ```
- Safetensors 원본: 630 tensors / 2 shards. `language_model.*` 191개(텍스트), `vision_tower.*` 437개, 기타(projector) 2개.

### 주요 소스 위치 (현재 워크트리: `/home/go/Workspace/llm_rs2-gemma3-4b`)
- `engine/src/models/config.rs` — `ModelConfig`, `RawHfConfig`, `detect_arch`, `from_json`
- `engine/src/models/mappers/mod.rs` — `WeightMapper` trait, `LayerWeightNames`, `create_mapper` factory
- `engine/src/models/mappers/gemma3.rs` — `Gemma3Mapper::weight_names`
- `engine/src/models/mappers/llama.rs`, `qwen2.rs` — 동일 패턴 (prefix 미도입 대상)
- `engine/src/models/loader/safetensors.rs` — `SafetensorsSource::open`, `resolve_name`, `load_raw`
- `engine/src/bin/generate.rs` — eval-ll 진입점 (`--eval-ll --eval-batch` 플래그)

### 주요 설계 제약
- `.cl` 커널 수정 금지 (이 계획에서는 커널 변경 없음).
- Release profile은 `lto = "fat"`. Debug/ASan 빌드는 별도 profile 사용.
- Android 빌드 시 `source android.source` 선행 필수. 4B(~8GB F16) smoke는 x86 호스트/CPU 백엔드에서 먼저 수행하는 것을 권장하되, OpenCL 타겟이 실제 검증 환경.
- 1B 로딩/테스트 회귀 금지 — 모든 변경은 1B 동작에 영향 없어야 함.

### 파일 구조 (변경 대상)

| 경로 | 역할 | 변경 종류 |
|------|------|-----------|
| `engine/src/models/config.rs` | Config 파싱 + `ModelConfig` 구조체 | 수정 — `weight_prefix` 필드, multimodal flatten, 테스트 추가 |
| `engine/src/models/mappers/mod.rs` | `WeightMapper` trait, factory | 수정 — `create_mapper(arch, prefix)`, trait 메서드에서 prefix 사용 |
| `engine/src/models/mappers/gemma3.rs` | Gemma3 tensor name 매핑 | 수정 — 구조체에 `prefix` 필드, 모든 format! 앞에 `{prefix}` 추가 |
| `engine/src/models/mappers/llama.rs` | Llama tensor name 매핑 | 수정 — 동일 패턴 적용(기본값 `""`) |
| `engine/src/models/mappers/qwen2.rs` | Qwen2 tensor name 매핑 | 수정 — 동일 패턴 적용(기본값 `""`) |
| `engine/src/models/loader/safetensors.rs` | safetensors 로더 | 수정 — `create_mapper(config.arch, &config.weight_prefix)` 호출, `load_raw` 에러 메시지 개선 |
| `engine/src/models/loader/gguf.rs` | GGUF 로더 | 수정 — 동일하게 `create_mapper(arch, "")` (GGUF는 prefix 없음) |
| `engine/tests/gemma3_4b_loading.rs` | 신규 통합 테스트 | 생성 — 실제 4B config/safetensors 로드 smoke |
| `engine/src/bin/generate.rs` (필요 시) | eval-ll 루프 | 조사 후 필요 시 수정 — 2번째 문항 크래시 원인 수정 |

---

## Phase 1: Baseline 재현과 회귀 기준 확립

### Task 1: 워크트리 상태 확인과 4B 크래시 재현

**Files:**
- 읽기만: `/home/go/Workspace/llm_rs2-gemma3-4b/`
- 실행 로그: `/tmp/gemma3_4b_baseline.log`

- [ ] **Step 1: 워크트리 및 브랜치 확인**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && git status && git branch --show-current`
Expected: clean, `feat/gemma3-4b-support`

- [ ] **Step 2: 현재 1B 테스트 스위트 통과 확인(회귀 기준)**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib models::`
Expected: `test_parse_llama_config`, `test_parse_gemma3_config`, `test_parse_qwen2_config`, `test_llama_mapper_names`, `test_qwen2_mapper_names`, `test_gemma3_mapper_names` 모두 PASS.

- [ ] **Step 3: 4B 원본 config와 safetensors 백업본 확인**

Run: `ls -la /home/go/Workspace/llm_rs2/models/gemma3-4b/ | head -40`
Expected: `config.json`, `config.json.orig`, `model.safetensors` (단일 파일) 존재.

- [ ] **Step 4: 4B 원본 config(`config.json.orig`) 내용을 `.orig` 포맷 그대로 확인**

Run: `head -60 /home/go/Workspace/llm_rs2/models/gemma3-4b/config.json.orig`
Expected: `"architectures": ["Gemma3ForConditionalGeneration"]`, `"text_config": {...}` 확인.

- [ ] **Step 5: 현재 워크어라운드 상태에서 eval-ll 크래시 재현(CPU 백엔드, smoke)**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
cargo build --release -p llm_rs2 --bin generate 2>&1 | tail -5
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend cpu --kv-type f32 --max-seq-len 4096 \
  --eval-ll --eval-batch /tmp/race_h_smoke_10q.json --greedy \
  2>&1 | tee /tmp/gemma3_4b_baseline.log | tail -40
```
Expected: 1번째 문항 처리 후 `free(): invalid size` 크래시(exit 134). 베이스라인 확보.

- [ ] **Step 6: Baseline 로그를 커밋용 주석 파일로 저장**

Run: `tail -20 /tmp/gemma3_4b_baseline.log > /tmp/gemma3_4b_baseline.tail.txt`
Expected: 크래시 직전 로그 20줄 저장.

- [ ] **Step 7: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git commit --allow-empty -m "chore(gemma3-4b): baseline — workaround config + crash on 2nd eval"
```

---

## Phase 2: Config + Mapper에 multimodal prefix/flatten 정식 지원

### Task 2: `RawHfConfig`에 multimodal 필드 추가 (실패 테스트 작성)

**Files:**
- Test: `engine/src/models/config.rs` (기존 `mod tests` 섹션에 추가)

- [ ] **Step 1: 실패 테스트 작성**

`engine/src/models/config.rs`의 `mod tests` 내부 맨 끝에 추가:
```rust
#[test]
fn test_parse_gemma3_multimodal_config_flattens_text_config() {
    let json = r#"{
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 10240,
            "vocab_size": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "query_pre_attn_scalar": 256,
            "model_type": "gemma3_text",
            "eos_token_id": 1
        },
        "vision_config": { "hidden_size": 1152 }
    }"#;
    let tmp_dir = std::path::PathBuf::from("/tmp/llm_rs2_test_gemma3_4b_config");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let config_path = tmp_dir.join("config.json");
    let mut f = std::fs::File::create(&config_path).unwrap();
    f.write_all(json.as_bytes()).unwrap();

    let config = ModelConfig::from_json(&tmp_dir).unwrap();
    assert_eq!(config.arch, ModelArch::Gemma3);
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 34);
    assert_eq!(config.num_attention_heads, 8);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.intermediate_size, 10240);
    assert_eq!(config.weight_prefix, "language_model.");
    assert_eq!(config.sliding_window, Some(1024));
    assert_eq!(config.query_pre_attn_scalar, Some(256));
}

#[test]
fn test_parse_gemma3_1b_has_empty_weight_prefix() {
    // 1B는 multimodal wrapper 아님 → prefix는 빈 문자열
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("models/gemma3-1b");
    if !dir.exists() {
        eprintln!("Skipping: model dir not found at {}", dir.display());
        return;
    }
    let config = ModelConfig::from_json(&dir).unwrap();
    assert_eq!(config.weight_prefix, "");
}

#[test]
fn test_parse_llama_has_empty_weight_prefix() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("models/llama3.2-1b");
    if !dir.exists() {
        eprintln!("Skipping: model dir not found at {}", dir.display());
        return;
    }
    let config = ModelConfig::from_json(&dir).unwrap();
    assert_eq!(config.weight_prefix, "");
}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib models::config::tests::test_parse_gemma3_multimodal_config_flattens_text_config test_parse_gemma3_1b_has_empty_weight_prefix test_parse_llama_has_empty_weight_prefix 2>&1 | tail -30`
Expected: 컴파일 오류 — `weight_prefix`는 존재하지 않음.

### Task 3: `ModelConfig`에 `weight_prefix` 추가 + `RawHfConfig`에 `text_config`/`architectures`

**Files:**
- Modify: `engine/src/models/config.rs`

- [ ] **Step 1: `ModelConfig` 구조체에 `weight_prefix` 필드 추가(17줄 부근)**

`pub struct ModelConfig { ... }` 내부에 `pub eos_token_id: u32,` 아래에 추가:
```rust
    /// Safetensors tensor name prefix (e.g., "language_model." for Gemma3 multimodal wrappers).
    /// Empty string for standard flat layouts (Llama, Qwen2, Gemma3 1B).
    pub weight_prefix: String,
```

- [ ] **Step 2: `RawHfConfig`에 `architectures`, `text_config` 필드 추가**

`struct RawHfConfig` 정의에서 `architectures: Option<Vec<String>>,`는 이미 존재. 맨 아래에 추가:
```rust
    /// Multimodal wrapper용 — text 전용 서브 config. Gemma3 4B의 "text_config"에 해당.
    /// 존재하면 from_json이 top-level 필드들을 여기서 가져온다.
    text_config: Option<Box<RawHfConfig>>,
```

- [ ] **Step 3: `from_json`에서 multimodal wrapper 감지 후 flatten + prefix 설정**

`pub fn from_json(model_dir: &Path) -> Result<Self>` 함수 내 `let raw: RawHfConfig = serde_json::from_reader(file)?;` 바로 아래에 삽입:
```rust
        // Multimodal wrapper 감지: text_config가 존재하거나 architectures가 *ForConditionalGeneration이면,
        // text_config를 top-level로 flatten하고 weight prefix를 설정한다.
        let is_multimodal = raw.architectures.as_ref().is_some_and(|archs| {
            archs.iter().any(|a| a.ends_with("ForConditionalGeneration"))
        }) || raw.text_config.is_some();

        let (raw, weight_prefix): (RawHfConfig, String) = if is_multimodal {
            let tc = *raw.text_config.clone().ok_or_else(|| {
                anyhow!("multimodal wrapper detected but text_config missing")
            })?;
            // text_config에 architectures가 없으면 상위의 값을 내부 causal LM 이름으로 치환해서 넘겨준다.
            let mut flat = tc;
            if flat.architectures.is_none() {
                // Gemma3ForConditionalGeneration → Gemma3ForCausalLM 으로 표준화
                flat.architectures = Some(vec!["Gemma3ForCausalLM".to_string()]);
            }
            // text_config 자신이 또 중첩되는 경우는 지원하지 않음 (단일 wrapper 한정)
            flat.text_config = None;
            (flat, "language_model.".to_string())
        } else {
            (raw, String::new())
        };
```

그리고 `Ok(Self { ... })` 블록 맨 아래(`embed_scale,` 바로 위)에 추가:
```rust
            weight_prefix,
```

`from_gguf_metadata`의 `Ok(Self { ... })` 블록 맨 아래에도 동일하게 추가:
```rust
            weight_prefix: String::new(),
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib models::config::tests 2>&1 | tail -20`
Expected: 새 3개 테스트 + 기존 3개 PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add engine/src/models/config.rs
git commit -m "feat(config): add weight_prefix and text_config flatten for multimodal wrappers"
```

### Task 4: `WeightMapper`에 prefix 주입 (실패 테스트)

**Files:**
- Test: `engine/src/models/mappers/mod.rs` (기존 `mod tests`에 추가)

- [ ] **Step 1: 실패 테스트 작성**

`mod tests` 내부에 추가:
```rust
    #[test]
    fn test_gemma3_mapper_with_multimodal_prefix() {
        let m = create_mapper_with_prefix(ModelArch::Gemma3, "language_model.");
        let names = m.weight_names(0);
        assert_eq!(names.wq, "language_model.model.layers.0.self_attn.q_proj.weight");
        assert_eq!(
            names.attention_norm,
            "language_model.model.layers.0.input_layernorm.weight"
        );
        assert_eq!(names.pre_ffn_norm.as_deref(),
            Some("language_model.model.layers.0.pre_feedforward_layernorm.weight"));
        assert_eq!(m.embed_name(), "language_model.model.embed_tokens.weight");
        assert_eq!(m.norm_name(), "language_model.model.norm.weight");
        // lm_head: 4B는 tie_word_embeddings=true이므로 파일에 없을 수 있으나, 이름은 prefix 적용.
        assert_eq!(m.lm_head_name(), "language_model.lm_head.weight");
    }

    #[test]
    fn test_mapper_empty_prefix_matches_default_factory() {
        let with = create_mapper_with_prefix(ModelArch::Llama, "");
        let default = create_mapper(ModelArch::Llama);
        assert_eq!(with.weight_names(3).wq, default.weight_names(3).wq);
        assert_eq!(with.embed_name(), default.embed_name());
    }
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib models::mappers::tests 2>&1 | tail -15`
Expected: `create_mapper_with_prefix` 미정의 오류.

### Task 5: Mapper factory + Gemma3/Llama/Qwen2 Mapper에 prefix 추가

**Files:**
- Modify: `engine/src/models/mappers/mod.rs`
- Modify: `engine/src/models/mappers/gemma3.rs`
- Modify: `engine/src/models/mappers/llama.rs`
- Modify: `engine/src/models/mappers/qwen2.rs`

- [ ] **Step 1: `WeightMapper` trait의 기본 embed/norm/lm_head 이름을 struct 메서드로 강제 이동**

`engine/src/models/mappers/mod.rs`의 trait 정의 수정. 기본 구현을 제거하고 각 Mapper가 구현하도록:
```rust
pub trait WeightMapper: Send + Sync {
    fn weight_names(&self, layer_idx: usize) -> LayerWeightNames;
    fn bias_names(&self, _layer_idx: usize) -> Option<LayerBiasNames> { None }
    fn embed_name(&self) -> String;
    fn norm_name(&self) -> String;
    fn lm_head_name(&self) -> String;
}
```

이유: prefix가 매퍼별 필드이므로 기본 구현으로 `&'static str` 반환 불가 — `String` 반환으로 변경.

- [ ] **Step 2: `create_mapper_with_prefix` factory 추가 + 기존 `create_mapper`는 빈 prefix wrapper**

`mod.rs`의 factory 영역을 아래로 교체:
```rust
pub fn create_mapper(arch: ModelArch) -> Box<dyn WeightMapper> {
    create_mapper_with_prefix(arch, "")
}

pub fn create_mapper_with_prefix(arch: ModelArch, prefix: &str) -> Box<dyn WeightMapper> {
    let p = prefix.to_string();
    match arch {
        ModelArch::Llama => Box::new(llama::LlamaMapper { prefix: p }),
        ModelArch::Qwen2 => Box::new(qwen2::Qwen2Mapper { prefix: p }),
        ModelArch::Gemma3 => Box::new(gemma3::Gemma3Mapper { prefix: p }),
    }
}
```

- [ ] **Step 3: `Gemma3Mapper`에 prefix 필드 추가하고 모든 format!에 `{prefix}` 삽입**

`engine/src/models/mappers/gemma3.rs` 전체를 아래로 교체:
```rust
use super::{LayerWeightNames, WeightMapper};

pub struct Gemma3Mapper {
    pub prefix: String,
}

impl WeightMapper for Gemma3Mapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        let p = &self.prefix;
        LayerWeightNames {
            wq: format!("{p}model.layers.{i}.self_attn.q_proj.weight"),
            wk: format!("{p}model.layers.{i}.self_attn.k_proj.weight"),
            wv: format!("{p}model.layers.{i}.self_attn.v_proj.weight"),
            wo: format!("{p}model.layers.{i}.self_attn.o_proj.weight"),
            w_gate: format!("{p}model.layers.{i}.mlp.gate_proj.weight"),
            w_up: format!("{p}model.layers.{i}.mlp.up_proj.weight"),
            w_down: format!("{p}model.layers.{i}.mlp.down_proj.weight"),
            attention_norm: format!("{p}model.layers.{i}.input_layernorm.weight"),
            ffn_norm: format!("{p}model.layers.{i}.post_attention_layernorm.weight"),
            pre_ffn_norm: Some(format!("{p}model.layers.{i}.pre_feedforward_layernorm.weight")),
            post_ffn_norm: Some(format!("{p}model.layers.{i}.post_feedforward_layernorm.weight")),
            q_norm: Some(format!("{p}model.layers.{i}.self_attn.q_norm.weight")),
            k_norm: Some(format!("{p}model.layers.{i}.self_attn.k_norm.weight")),
        }
    }
    fn embed_name(&self) -> String { format!("{}model.embed_tokens.weight", self.prefix) }
    fn norm_name(&self) -> String { format!("{}model.norm.weight", self.prefix) }
    fn lm_head_name(&self) -> String { format!("{}lm_head.weight", self.prefix) }
}
```

- [ ] **Step 4: `LlamaMapper`에 prefix 필드 적용**

`engine/src/models/mappers/llama.rs` 읽어서 같은 패턴으로 수정. 예시(구조):
```rust
use super::{LayerWeightNames, WeightMapper};

pub struct LlamaMapper {
    pub prefix: String,
}

impl WeightMapper for LlamaMapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        let p = &self.prefix;
        LayerWeightNames {
            wq: format!("{p}model.layers.{i}.self_attn.q_proj.weight"),
            wk: format!("{p}model.layers.{i}.self_attn.k_proj.weight"),
            wv: format!("{p}model.layers.{i}.self_attn.v_proj.weight"),
            wo: format!("{p}model.layers.{i}.self_attn.o_proj.weight"),
            w_gate: format!("{p}model.layers.{i}.mlp.gate_proj.weight"),
            w_up: format!("{p}model.layers.{i}.mlp.up_proj.weight"),
            w_down: format!("{p}model.layers.{i}.mlp.down_proj.weight"),
            attention_norm: format!("{p}model.layers.{i}.input_layernorm.weight"),
            ffn_norm: format!("{p}model.layers.{i}.post_attention_layernorm.weight"),
            pre_ffn_norm: None,
            post_ffn_norm: None,
            q_norm: None,
            k_norm: None,
        }
    }
    fn embed_name(&self) -> String { format!("{}model.embed_tokens.weight", self.prefix) }
    fn norm_name(&self) -> String { format!("{}model.norm.weight", self.prefix) }
    fn lm_head_name(&self) -> String { format!("{}lm_head.weight", self.prefix) }
}
```

- [ ] **Step 5: `Qwen2Mapper`에 prefix 필드 적용**

`engine/src/models/mappers/qwen2.rs`도 동일 패턴 — struct에 `prefix: String` 필드 추가, `weight_names`의 모든 format!에 `{p}` 삽입, `bias_names`의 format!에도 `{p}` 삽입, `embed_name`/`norm_name`/`lm_head_name`을 `String` 반환으로 변경.

- [ ] **Step 6: Mapper 테스트 통과 확인**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib models::mappers 2>&1 | tail -20`
Expected: 기존 3개 테스트 + 새 2개 PASS (총 5개).

### Task 6: `SafetensorsSource`와 `gguf` 로더에서 새 factory 호출

**Files:**
- Modify: `engine/src/models/loader/safetensors.rs` (58줄 부근)
- Modify: `engine/src/models/loader/gguf.rs`

- [ ] **Step 1: `safetensors.rs`의 `create_mapper` 호출을 `create_mapper_with_prefix`로 변경**

`engine/src/models/loader/safetensors.rs` 22~30줄 근방의 import 문 수정:
```rust
use crate::models::mappers::{WeightMapper, create_mapper_with_prefix};
```

58줄 부근을 수정:
```rust
        let mapper = create_mapper_with_prefix(config.arch, &config.weight_prefix);
```

- [ ] **Step 2: `resolve_name` 이후 이름이 `&str`에서 `String`을 받도록 변경 (trait 반환 타입 변경 대응)**

`resolve_name` 내부의 `TensorId::Embed`/`FinalNorm`/`LmHead` 분기에서 `.to_string()` 제거:
```rust
            TensorId::Embed => self.mapper.embed_name(),
            TensorId::FinalNorm => self.mapper.norm_name(),
            TensorId::LmHead => self.mapper.lm_head_name(),
```

- [ ] **Step 3: `load_raw`의 missing tensor 에러 메시지 개선**

`fn load_raw(...)` 내부 178~184줄 부근(`self.shard_tensors[shard_idx].tensor(name)` 오류 분기)을 수정:
```rust
            Err(e) => {
                // 진단 정보: 어느 shard에 어떤 이름이 있는지 앞뒤 몇 개만 기록
                let sample: Vec<String> = self.shard_tensors[shard_idx]
                    .names().iter().take(5).map(|s| (*s).to_string()).collect();
                return Err(anyhow!(
                    "safetensors missing tensor '{name}' in shard {shard_idx} (prefix={:?}); first few names: {sample:?}; err={e}",
                    self.config.weight_prefix
                ));
            }
```

- [ ] **Step 4: `gguf.rs`의 mapper 호출도 새 factory 시그니처에 맞춤**

`engine/src/models/loader/gguf.rs`에서 `create_mapper(arch)` 또는 기본 factory를 호출하는 지점이 있다면 `create_mapper_with_prefix(arch, "")`로 바꾸거나, 기존 `create_mapper`를 그대로 사용(이미 빈 prefix wrapper로 만들었음). 빌드 에러만 없으면 그대로 둔다.

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo check -p llm_rs2 2>&1 | tail -20`
Expected: 오류 없음. 오류 있으면 해당 지점을 `create_mapper_with_prefix`로 전환.

- [ ] **Step 5: 전체 유닛 테스트 실행**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib 2>&1 | tail -20`
Expected: 283개 이상 PASS, 회귀 없음.

- [ ] **Step 6: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add engine/src/models/mappers/ engine/src/models/loader/safetensors.rs engine/src/models/loader/gguf.rs
git commit -m "feat(mappers): prefix-aware WeightMapper for Gemma3 multimodal safetensors"
```

### Task 7: 원본 4B config/safetensors로 통합 smoke 테스트

**Files:**
- Create: `engine/tests/gemma3_4b_loading.rs`
- 모델 경로 복원: `/home/go/Workspace/llm_rs2/models/gemma3-4b/config.json`에 `config.json.orig`을 덮어쓰고(원본 사용), 4B 원본 multi-shard safetensors를 확보한다.

- [ ] **Step 1: 원본 config 복원(백업 먼저)**

Run:
```bash
cd /home/go/Workspace/llm_rs2/models/gemma3-4b
cp config.json config.json.flattened.bak
cp config.json.orig config.json
ls -la config.json config.json.orig config.json.flattened.bak
```
Expected: 원본이 `config.json` 위치에 복원.

- [ ] **Step 2: 원본 safetensors 확보**

Run:
```bash
ls -la /home/go/Workspace/llm_rs2/models/gemma3-4b/*.safetensors* 2>&1
```
상태에 따라 분기:
- 원본 2-shard 파일이 그대로 있다면(예: `model-00001-of-00002.safetensors`) 그대로 사용.
- 워크어라운드로 prefix-stripped 단일 파일만 남아있다면, HuggingFace에서 원본 2-shard로 재다운로드 필요:
  ```bash
  # 선행조건: HF_TOKEN 환경변수 세팅, gemma-3-4b-pt 게이트 승인 필요
  huggingface-cli download google/gemma-3-4b-pt \
    --local-dir /home/go/Workspace/llm_rs2/models/gemma3-4b-orig \
    --include "*.safetensors" "*.safetensors.index.json" "config.json"
  ```
  그 후 `models/gemma3-4b-orig`를 이 단계부터 사용한다(아래 `MODEL_DIR` 변경).

- [ ] **Step 3: 통합 smoke 테스트 작성**

`engine/tests/gemma3_4b_loading.rs`:
```rust
//! Gemma 3 4B multimodal wrapper loading smoke test.
//! Skips silently if the 4B model directory is unavailable.

use std::path::Path;

const MODEL_DIR_ENV: &str = "LLM_RS2_GEMMA3_4B_DIR";
const DEFAULT_DIR: &str = "/home/go/Workspace/llm_rs2/models/gemma3-4b";

#[test]
fn gemma3_4b_config_flattens_and_sets_prefix() {
    let dir = std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| DEFAULT_DIR.to_string());
    let p = Path::new(&dir);
    if !p.join("config.json").exists() {
        eprintln!("Skipping: {}/config.json not found", dir);
        return;
    }
    let cfg = llm_rs2::models::config::ModelConfig::from_json(p).unwrap();
    assert_eq!(cfg.arch, llm_rs2::models::config::ModelArch::Gemma3);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 34);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim, 256);
    // flat config (워크어라운드) vs. 원본 wrapper 둘 다 허용 — 원본이면 prefix가 설정됨.
    if cfg.weight_prefix.is_empty() {
        eprintln!("note: running against flattened workaround config.json");
    } else {
        assert_eq!(cfg.weight_prefix, "language_model.");
    }
}

#[test]
fn gemma3_4b_safetensors_resolves_layer0_qproj() {
    let dir = std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| DEFAULT_DIR.to_string());
    let p = Path::new(&dir);
    if !p.join("config.json").exists() {
        eprintln!("Skipping: {}/config.json not found", dir);
        return;
    }
    let src = llm_rs2::models::loader::safetensors::SafetensorsSource::open(
        &dir, llm_rs2::core::DType::F16
    ).unwrap();
    let name = src.resolve_name(&llm_rs2::models::loader::TensorId::LayerWeight {
        layer: 0,
        kind: llm_rs2::models::loader::LayerWeightKind::Wq,
    });
    // 원본 wrapper인지 flattened wrapper인지에 따라 prefix 달라짐.
    assert!(name.ends_with("model.layers.0.self_attn.q_proj.weight"), "got: {name}");
}
```

(실제 `pub use` 경로가 다르면 컴파일 오류를 보고 수정한다. `engine/src/lib.rs`의 re-export를 확인.)

- [ ] **Step 4: 테스트 실행**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test --test gemma3_4b_loading 2>&1 | tail -20`
Expected: 2개 PASS. 실패 시 오류 메시지를 기반으로 수정.

- [ ] **Step 5: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add engine/tests/gemma3_4b_loading.rs
git commit -m "test(gemma3-4b): integration smoke for multimodal config + mapper"
```

---

## Phase 3: 4B eval-ll 2번째 문항 크래시 근본 원인 해결

### Task 8: 정식 로딩 경로에서 크래시 재현/사라짐 확인

**Files:**
- Read-only: `engine/src/bin/generate.rs` (eval-ll 루프 위치 파악)
- 실행 로그: `/tmp/gemma3_4b_after_phase2.log`

- [ ] **Step 1: 원본 4B 디렉터리로 eval-ll smoke 재실행**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
cargo build --release -p llm_rs2 --bin generate 2>&1 | tail -3
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend cpu --kv-type f32 --max-seq-len 4096 \
  --eval-ll --eval-batch /tmp/race_h_smoke_10q.json --greedy \
  2>&1 | tee /tmp/gemma3_4b_after_phase2.log | tail -40
```

- [ ] **Step 2: 결과에 따라 분기**

- (A) 크래시 **사라짐** (10/10 정상 처리) → 원인은 prefix strip 워크어라운드가 누락/중복 텐서를 만들어 버퍼 레이아웃을 깼던 것. Phase 3은 `Task 10`(검증)으로 바로 이동.
- (B) 크래시 **여전** → Task 9(디버깅)로 진행.

- [ ] **Step 3: 중간 결과 기록 커밋**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
echo "== Phase2 후 smoke 결과 ==" > notes/gemma3_4b_phase3_status.md
date >> notes/gemma3_4b_phase3_status.md
tail -30 /tmp/gemma3_4b_after_phase2.log >> notes/gemma3_4b_phase3_status.md
mkdir -p notes
git add notes/gemma3_4b_phase3_status.md 2>/dev/null || true
git commit -m "chore(gemma3-4b): Phase 3 baseline — smoke result after Phase 2" || true
```

### Task 9: (크래시 재현 시) AddressSanitizer/진단 경로로 원인 국소화

**Files:**
- Read-only: `engine/src/bin/generate.rs` eval-ll 루프 근처
- Modify(디버깅용 임시): `engine/src/core/kv_cache.rs` 또는 `engine/src/models/llama/llama_model.rs` 진단 로그 한정 추가

- [ ] **Step 1: eval-ll 루프에서 문항 간에 일어나는 작업을 파악**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
grep -n "eval_ll\|eval-ll\|eval_batch" engine/src/bin/generate.rs | head -30
```
Expected: eval_ll 함수/블록 위치 확인.

- [ ] **Step 2: 2번째 문항 진입 직전까지의 할당/해제 경로 특정 후보**

후보 체크리스트(읽기 + 추론만):
- KV cache reset/재할당: `CacheManager`가 문항 사이에 `force_evict`/`reset` 하는가?
- workspace buffer: `LayerWorkspace`가 문항 사이에 재할당되는가, 같은 size로 재사용되는가?
- tokenizer의 vocab_size=262144에 비례한 logits buffer — 4B에서 256000 이상? (`vocab_size` 확인)
- F32 KV 경로에서 `--kv-type f32` + 4096 seq + 34 layer + kv_heads=4 + head_dim=256 = per-layer 16MB × 34 = 544MB. 시스템 메모리 확인.

- [ ] **Step 3: ASan 빌드용 별도 Cargo 프로필로 재빌드**

`/home/go/Workspace/llm_rs2-gemma3-4b/.cargo/config.toml` 기존 설정을 보존하면서 임시 환경변수로 ASan 빌드:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
RUSTFLAGS="-Z sanitizer=address" \
  cargo +nightly build -p llm_rs2 --bin generate --target x86_64-unknown-linux-gnu \
  --no-default-features --features cpu 2>&1 | tail -5
```
만약 `opencl`이 기본 feature라서 ASan 빌드가 실패하면 features 조정. OpenCL 없이 CPU 백엔드만 빌드해야 ASan 안정.

- [ ] **Step 4: ASan 실행으로 스택 추적 획득**

Run:
```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
./target/x86_64-unknown-linux-gnu/debug/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend cpu --kv-type f32 --max-seq-len 4096 \
  --eval-ll --eval-batch /tmp/race_h_smoke_10q.json --greedy \
  2>&1 | tee /tmp/gemma3_4b_asan.log | tail -80
```
Expected: heap-buffer-overflow / double-free의 소스 라인 확보.

- [ ] **Step 5: 국소화된 원인에 정밀 수정**

수정 지점은 ASan 결과에 의존. 원칙:
- 가설 기반 수정을 피하고, ASan 스택에 명시된 라인만 수정.
- 수정 후 ASan 빌드에서 2번째~10번째 문항까지 정상 통과하는지 재확인.
- fix별로 별도 커밋.

- [ ] **Step 6: Commit (수정이 있을 때마다)**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add <fix 파일들>
git commit -m "fix(<scope>): <ASan-localized crash cause — one sentence>"
```

### Task 10: OpenCL 백엔드 + 최종 eval-ll 검증

**Files:**
- 실행 로그: `/tmp/gemma3_4b_opencl_evalll.log`

- [ ] **Step 1: OpenCL 빌드 및 eval-ll 재현**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
cargo build --release -p llm_rs2 --bin generate 2>&1 | tail -3
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both \
  --eval-ll --eval-batch /tmp/race_h_smoke_10q.json --greedy \
  2>&1 | tee /tmp/gemma3_4b_opencl_evalll.log | tail -40
```
Expected: 10/10 문항 정상 처리, `[Eval-LL] 10/10 ...` 완료 라인.

- [ ] **Step 2: 간단한 single-prompt 생성 smoke도 수행**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 512 \
  --prompt "The capital of France is" --max-tokens 16 --greedy \
  2>&1 | tail -20
```
Expected: 합리적인 text 완성 ("Paris").

- [ ] **Step 3: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add .  # notes 등 포함 시
git commit -m "chore(gemma3-4b): eval-ll smoke green on OpenCL" || true
```

---

## Phase 4: 최종 검증 — cargo 풀 스위트 + QCF 스모크 + 회귀

### Task 11: cargo fmt + clippy + full test

**Files:** 없음 (sanity-check 스킬 사용 권장)

- [ ] **Step 1: sanity-check**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo fmt --all && cargo clippy -p llm_rs2 --all-targets -- -D warnings 2>&1 | tail -10`
Expected: warning 0.

- [ ] **Step 2: 전체 unit + integration**

Run: `cd /home/go/Workspace/llm_rs2-gemma3-4b && cargo test -p llm_rs2 --lib --tests 2>&1 | tail -10`
Expected: 283+ PASS, 회귀 없음.

- [ ] **Step 3: 1B 회귀 — Llama 3.2 1B / Gemma 3 1B / Qwen2 1.5B smoke 한 번씩**

Run(각 경로에 모델이 존재할 때):
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
for m in llama3.2-1b gemma3-1b qwen2.5-1.5b; do
  DIR=/home/go/Workspace/llm_rs2/models/$m
  [ -d "$DIR" ] || { echo "skip $m"; continue; }
  ./target/release/generate \
    --model-path "$DIR" --backend cpu --max-seq-len 256 \
    --prompt "Hello," --max-tokens 8 --greedy 2>&1 | tail -3
done
```
Expected: 3개 모델 모두 정상 토큰 생성.

### Task 12: PACT QCF 스모크 파이프라인과 연결

**Files:** 없음

- [ ] **Step 1: QCF both 모드 smoke (10문항)**

Run: Task 10 Step 1과 동일 커맨드. 이미 성공했으면 스킵 가능.

- [ ] **Step 2: race_h 전체(또는 부분) 배치 실행**

Run:
```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
BATCH=/home/go/Workspace/papers/pact2026/experiments/.../race_h.json  # 실제 경로
[ -f "$BATCH" ] || { echo "skip: batch not found"; exit 0; }
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll --eval-batch "$BATCH" --greedy \
  2>&1 | tail -20
```
Expected: 배치 완료, 결과 파일 저장.

- [ ] **Step 3: 최종 Commit + PR 브랜치 준비**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git log --oneline master..HEAD
git push -u origin feat/gemma3-4b-support  # (사용자 승인 필요)
```
Expected: 커밋 체인 확인. push는 사용자 확인 후.

### Task 13: 원본 config/safetensors 위치 정리 및 문서화

**Files:**
- Modify: `CLAUDE.md` 또는 `docs/` 내 모델 목록 파일(존재 시)

- [ ] **Step 1: 모델 디렉터리 상태 정리**

Run:
```bash
cd /home/go/Workspace/llm_rs2/models/gemma3-4b
ls -la
# config.json.flattened.bak는 보존(회귀용), config.json은 원본 유지 — 정식 경로가 원본을 기대하므로.
```

- [ ] **Step 2: 간단한 supported model 목록 업데이트 (있는 경우에만)**

`docs/` 또는 README에 모델 목록이 있다면 "Gemma 3 4B (multimodal wrapper, text-only path)" 한 줄 추가. 없으면 스킵.

- [ ] **Step 3: 데스크톱 알림**

Run: `notify-send "llm.rs" "Gemma 3 4B support — implementation plan complete"`

- [ ] **Step 4: Commit**

```bash
cd /home/go/Workspace/llm_rs2-gemma3-4b
git add -A
git commit -m "docs(gemma3-4b): note multimodal wrapper support in model list" || true
```

---

## 완료 기준 (Definition of Done)

1. 원본(`config.json.orig`) 구조의 Gemma 3 4B config를 그대로 로드 가능 — `text_config` auto-flatten + `weight_prefix="language_model."` 설정.
2. 원본 2-shard safetensors에서 `language_model.` prefix 텐서를 매퍼가 정확히 resolve.
3. `--eval-ll --eval-batch` 10문항 배치 실행 시 크래시 없이 10/10 완료.
4. 1B 3개 모델(Llama/Gemma3/Qwen2) 회귀 없음, 유닛 테스트 283+ 통과.
5. `cargo fmt` clean, `cargo clippy -- -D warnings` clean.
6. 모든 커밋 conventional commits 규칙 준수, 논리 단위로 분리.

## 리스크 & 대응

- **R1: 4B 원본 safetensors가 로컬에 없다면** — Task 7 Step 2의 `huggingface-cli` 다운로드 필요, `google/gemma-3-4b-pt` gated repo → HF 토큰 + 승인 필요. 대안으로 기존 워크어라운드 파일을 prefix 없이도 로드 가능한 상태로 유지하고 "flattened workaround" 경로도 계속 지원 (테스트는 양쪽 모두 skip-safe).
- **R2: Phase 3 크래시가 ASan에서도 재현 안 될 수 있음** — CPU/OpenCL 동작 차이. OpenCL 경로에서만 재현되면 `RUSTFLAGS="-C debug-assertions=on"` + `OPENCL_ENABLE_DEBUG=1` 로 로그 보강 후 분기.
- **R3: Trait 반환 타입 변경(`&str → String`)에 의한 호출처 광범위 영향** — Task 6 Step 4에서 `cargo check`로 전체 빌드 에러를 훑은 뒤, 각 호출처에서 `.to_string()` 제거 / 새 `String` 적용.
- **R4: PACT 마감 4/24** — Phase 2까지만 완료해도 워크어라운드 모델로 돌아감(현 상태와 동일). Phase 3 크래시가 prefix 문제였다면 Phase 2로 자동 해결. 아니라면 Phase 3에 별도 세션/서브에이전트 집중 투입.
