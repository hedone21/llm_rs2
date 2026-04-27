/// auf-tool — AUF (Argus Unified Format) 자산 빌드/검사/수정/검증 CLI 도구.
///
/// ENG-ALG-223 §3.12.17 구현. WSWAP-3.7B-CLI.
///
/// 서브커맨드:
///   build  — GGUF + tokenizer.json → AUF (사전 SOA + Q/K permute 변환)
///   info   — AUF 헤더/섹션/메타 출력
///   strip  — dead variant 섹션 제거 (selective, 백업 옵션)
///   verify — magic/section/required capability/offset 무결성 전수 검증
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};

use llm_rs2::auf::stripper::strip;
use llm_rs2::auf::tensor_index::{
    LAYER_IDX_CROSS, TensorDType, TensorEntry, TensorIndex, TensorKind,
};
use llm_rs2::auf::{
    AufError, AufMeta, AufTokenizer, AufWriter, BackendTag, SECTION_STRIPPABLE,
    TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS, TAG_WEIGHTS_CUDA_AOS, TOKENIZER_KIND_BPE,
    compute_source_hash, open, q4_0_aos_to_adreno_soa,
};

// ---------------------------------------------------------------------------
// CLI 정의 (clap derive)
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "auf-tool",
    about = "AUF (Argus Unified Format) 자산 빌드/검사/수정/검증 도구",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// GGUF + tokenizer.json → AUF 파일 빌드
    Build(BuildArgs),
    /// AUF 헤더/섹션/메타 정보 출력
    Info(InfoArgs),
    /// dead variant 섹션 제거 (selective strip)
    Strip(StripArgs),
    /// AUF 무결성 전수 검증
    Verify(VerifyArgs),
}

// ── build ──────────────────────────────────────────────────────────────────

#[derive(Args)]
struct BuildArgs {
    /// 원본 GGUF 파일 경로
    #[arg(long, value_name = "FILE")]
    input: PathBuf,

    /// tokenizer.json 경로
    #[arg(long, value_name = "FILE")]
    tokenizer: PathBuf,

    /// 출력 AUF 파일 경로 (atomic rename으로 부분 write 방지)
    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    /// 포함할 backend variant (comma-separated 또는 "all").
    /// 예: "adreno_soa,cpu_aos" 또는 "WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS" 또는 "all"
    #[arg(long, value_name = "VARIANTS")]
    variants: String,

    /// 헤더 created_by 문자열 (최대 32B UTF-8). 기본: "llm_rs2 auf-tool v<VERSION>"
    #[arg(long, value_name = "STRING")]
    created_by: Option<String>,

    /// lm_head Q4_0 사전 변환 (v0.1.1, Sprint G-1).
    ///
    /// - `auto` (기본값): GGUF lm_head dtype != Q4_0이면 quantize, 이미 Q4_0이면
    ///   AOS bytes를 entry로 그대로 동봉.
    /// - `on`: 강제 quantize (이미 Q4_0이면 AOS bytes 동봉).
    /// - `off`: lm_head Q4_0 entry 미포함 (v0.1.0 byte-level 호환 출력,
    ///   capability bit 2 = 0, format_patch = 0).
    #[arg(long, value_name = "MODE", default_value = "auto")]
    include_lm_head: String,

    /// 진행 로그 출력 억제
    #[arg(long)]
    quiet: bool,
}

/// `--include-lm-head` 모드.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncludeLmHeadMode {
    /// lm_head Q4_0 entry 미포함 (v0.1.0 호환).
    Off,
    /// 강제 포함 (이미 Q4_0이면 그대로 동봉, F16/F32 → Q4_0 quantize).
    On,
    /// auto: GGUF lm_head dtype에 따라. 비-Q4_0 → quantize, Q4_0 → 그대로 동봉.
    Auto,
}

impl IncludeLmHeadMode {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(IncludeLmHeadMode::Auto),
            "on" | "true" | "yes" | "1" => Ok(IncludeLmHeadMode::On),
            "off" | "false" | "no" | "0" => Ok(IncludeLmHeadMode::Off),
            other => bail!(
                "Invalid --include-lm-head '{}'. Valid: auto, on, off",
                other
            ),
        }
    }
}

// ── info ──────────────────────────────────────────────────────────────────

#[derive(Args)]
struct InfoArgs {
    /// AUF 파일 경로
    #[arg(value_name = "FILE")]
    file: PathBuf,
}

// ── strip ─────────────────────────────────────────────────────────────────

#[derive(Args)]
struct StripArgs {
    /// AUF 파일 경로 (in-place 수정)
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// 유지할 section tag (comma-separated).
    /// required section (META/TOKENIZER/TENSOR_INDEX)은 자동 보존.
    /// 예: "WEIGHTS_ADRENO_SOA" 또는 "adreno_soa"
    #[arg(long, value_name = "TAGS")]
    keep: String,

    /// 백업 파일 생성 생략 (기본: <file>.bak 자동 생성)
    #[arg(long)]
    no_backup: bool,
}

// ── verify ────────────────────────────────────────────────────────────────

#[derive(Args)]
struct VerifyArgs {
    /// AUF 파일 경로
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// source GGUF 경로 (제공 시 source_hash 재계산하여 비교)
    #[arg(long, value_name = "FILE")]
    source: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Build(args) => cmd_build(args),
        Commands::Info(args) => cmd_info(args),
        Commands::Strip(args) => cmd_strip(args),
        Commands::Verify(args) => cmd_verify(args),
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// build 커맨드
// ---------------------------------------------------------------------------

/// variant 문자열 → section tag 목록으로 변환.
///
/// 입력: "all", "adreno_soa,cpu_aos", "WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS" 등 혼합 허용.
fn parse_variants(variants_str: &str) -> Result<Vec<&'static str>> {
    let s = variants_str.trim();
    if s.eq_ignore_ascii_case("all") {
        return Ok(vec![
            TAG_WEIGHTS_ADRENO_SOA,
            TAG_WEIGHTS_CUDA_AOS,
            TAG_WEIGHTS_CPU_AOS,
        ]);
    }
    let mut tags = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        let tag = match part.to_ascii_uppercase().as_str() {
            "WEIGHTS_ADRENO_SOA" | "ADRENO_SOA" | "ADRENO" => TAG_WEIGHTS_ADRENO_SOA,
            "WEIGHTS_CUDA_AOS" | "CUDA_AOS" | "CUDA" => TAG_WEIGHTS_CUDA_AOS,
            "WEIGHTS_CPU_AOS" | "CPU_AOS" | "CPU" => TAG_WEIGHTS_CPU_AOS,
            other => bail!(
                "Unknown variant '{}'. Valid: adreno_soa, cuda_aos, cpu_aos, all",
                other
            ),
        };
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    if tags.is_empty() {
        bail!("--variants는 하나 이상의 variant를 지정해야 합니다");
    }
    Ok(tags)
}

/// tokenizer.json에서 AufTokenizer를 구성한다.
///
/// tokenizers crate의 Tokenizer::from_file은 내부 vocab/merges 직접 접근이
/// 어려우므로 JSON을 직접 파싱하여 필드를 추출한다.
fn load_tokenizer_from_json(path: &std::path::Path) -> Result<AufTokenizer> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Cannot open tokenizer.json {}: {}", path.display(), e))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|e| anyhow!("Cannot read tokenizer.json: {}", e))?;

    let json: serde_json::Value =
        serde_json::from_str(&buf).map_err(|e| anyhow!("Invalid tokenizer.json JSON: {}", e))?;

    let model = json
        .get("model")
        .ok_or_else(|| anyhow!("tokenizer.json: missing 'model' field"))?;

    // vocab: { "token_string": id, ... }
    let vocab_obj = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("tokenizer.json: missing model.vocab"))?;

    let vocab_size = vocab_obj.len();
    // id → bytes 매핑 (id 순서로 정렬)
    let mut id_to_token: Vec<(u64, Vec<u8>)> = vocab_obj
        .iter()
        .map(|(k, v)| {
            let id = v.as_u64().unwrap_or(0);
            (id, k.as_bytes().to_vec())
        })
        .collect();
    id_to_token.sort_by_key(|(id, _)| *id);

    // added_tokens에서 ID > vocab_size 인 토큰 추가 (Llama special tokens)
    let mut max_id = id_to_token.last().map(|(id, _)| *id).unwrap_or(0);
    if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for tok in added_tokens {
            let id = tok.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
            let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if id > max_id {
                // added_tokens에만 있는 token은 gap을 빈 토큰으로 채워야 함
                while max_id + 1 < id {
                    max_id += 1;
                    id_to_token.push((max_id, format!("<pad_{}>", max_id).into_bytes()));
                }
                id_to_token.push((id, content.as_bytes().to_vec()));
                max_id = id;
            }
        }
    }
    id_to_token.sort_by_key(|(id, _)| *id);
    let tokens: Vec<Vec<u8>> = id_to_token.into_iter().map(|(_, t)| t).collect();
    let vocab_size_final = tokens.len();

    // merges: ["Ġ t", "Ġ a", ...] 형태
    let merges: Vec<String> = model
        .get("merges")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                .collect()
        })
        .unwrap_or_default();

    // special token IDs (bos, eos 등)
    // tokenizer.json의 added_tokens에서 special=true인 것들을 키워드로 분류
    let find_special_id = |keywords: &[&str]| -> i32 {
        json.get("added_tokens")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter().find(|tok| {
                    let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    keywords.iter().any(|kw| content.contains(kw))
                })
            })
            .and_then(|tok| tok.get("id").and_then(|v| v.as_i64()))
            .map(|id| id as i32)
            .unwrap_or(-1)
    };

    let bos_id = find_special_id(&["begin_of_text", "bos", "<s>", "[BOS]"]);
    let eos_id = find_special_id(&["end_of_text", "eos", "</s>", "<|eot_id|>", "[EOS]"]);
    let pad_id = find_special_id(&["pad", "[PAD]", "<pad>"]);
    let unk_id = find_special_id(&["unk", "<unk>", "[UNK]"]);

    // chat_template (선택)
    let chat_template = json
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());

    eprintln!(
        "[auf-tool] Tokenizer: vocab={}, merges={}, bos={}, eos={}, chat_template={}",
        vocab_size_final,
        merges.len(),
        bos_id,
        eos_id,
        chat_template.is_some()
    );
    let _ = vocab_size; // suppress warning

    Ok(AufTokenizer {
        kind: TOKENIZER_KIND_BPE,
        tokens,
        merges,
        bos_id,
        eos_id,
        pad_id,
        unk_id,
        chat_template,
    })
}

/// GGUF에서 AufMeta를 구성한다.
fn build_meta_from_gguf(gguf: &llm_rs2::models::loader::gguf::GgufFile) -> Result<AufMeta> {
    use llm_rs2::models::config::ModelConfig;

    let config = ModelConfig::from_gguf_metadata(gguf)
        .map_err(|e| anyhow!("ModelConfig 파싱 실패: {}", e))?;

    // max_seq_len: ModelConfig에 없으므로 GGUF에서 직접 읽는다
    let arch_str = gguf.get_str("general.architecture").unwrap_or("llama");
    let max_seq_len = gguf
        .get_u32(&format!("{arch_str}.context_length"))
        .unwrap_or(2048);

    Ok(AufMeta {
        architecture: format!("{:?}", config.arch).to_lowercase(),
        n_layers: config.num_hidden_layers as u32,
        n_heads_q: config.num_attention_heads as u32,
        n_kv_heads: config.num_key_value_heads as u32,
        head_dim: config.head_dim as u32,
        hidden_dim: config.hidden_size as u32,
        ffn_dim: config.intermediate_size as u32,
        vocab_size: config.vocab_size as u32,
        max_seq_len,
        rope_theta: config.rope_theta,
        rotary_dim: config.head_dim as u32,
        rope_scaling: 1.0,
        rms_norm_epsilon: config.rms_norm_eps,
        default_dtype: None, // v0.2 multi-dtype에서만 설정 (Sprint C에서 확장)
    })
}

/// 레거시 호환을 위한 단순 분리 (transpose / unshuffle 없음). 새 코드에서는
/// `llm_rs2::auf::q4_0_aos_to_adreno_soa`를 사용하라. 일부 단위 테스트가 이
/// 헬퍼의 경계 동작을 그대로 사용하므로 보존한다.
#[allow(dead_code)]
fn q4_0_aos_to_soa(blocks: &[u8]) -> (Vec<u8>, Vec<u8>) {
    const BLOCK_SIZE: usize = 18;
    let n_blocks = blocks.len() / BLOCK_SIZE;
    let mut q_buf = Vec::with_capacity(n_blocks * 16);
    let mut d_buf = Vec::with_capacity(n_blocks * 2);
    for i in 0..n_blocks {
        let off = i * BLOCK_SIZE;
        d_buf.extend_from_slice(&blocks[off..off + 2]);
        q_buf.extend_from_slice(&blocks[off + 2..off + 18]);
    }
    (q_buf, d_buf)
}

/// Q4_0 AOS 바이트에 64B 또는 128B 패딩(align)을 추가하여 반환한다.
///
/// CPU AOS: 64B align, CUDA AOS: 128B align.
/// 사전 Q/K permute가 적용된 bytes를 그대로 정렬 패딩만 추가한다.
fn q4_0_aos_with_align(blocks: &[u8], alignment: usize) -> Vec<u8> {
    let size = blocks.len();
    let padded = size.div_ceil(alignment) * alignment;
    let mut out = vec![0u8; padded];
    out[..size].copy_from_slice(blocks);
    out
}

/// build 커맨드: GGUF → AUF
fn cmd_build(args: BuildArgs) -> Result<()> {
    // 1) variant 파싱
    let variant_tags = parse_variants(&args.variants)?;
    let lm_head_mode = IncludeLmHeadMode::parse(&args.include_lm_head)?;
    let quiet = args.quiet;

    if !quiet {
        eprintln!(
            "[auf-tool] Build: {} → {}",
            args.input.display(),
            args.output.display()
        );
        eprintln!("[auf-tool] Variants: {:?}", variant_tags);
        eprintln!("[auf-tool] include_lm_head: {:?}", lm_head_mode);
    }

    // 2) source_hash 계산
    if !quiet {
        eprint!("[auf-tool] Computing source_hash...");
    }
    let (source_hash, source_size, source_mtime) =
        compute_source_hash(&args.input).map_err(|e| anyhow!("source_hash 계산 실패: {}", e))?;
    if !quiet {
        eprintln!(" done ({} bytes)", source_size);
    }

    // 3) GGUF 파싱
    if !quiet {
        eprint!("[auf-tool] Parsing GGUF...");
    }
    let gguf = llm_rs2::models::loader::gguf::GgufFile::open(&args.input)
        .map_err(|e| anyhow!("GGUF 파일 열기 실패: {}", e))?;
    if !quiet {
        eprintln!(" {} tensors", gguf.tensors.len());
    }

    // 4) AufMeta 구성
    let meta =
        build_meta_from_gguf(&gguf).map_err(|e| anyhow!("GGUF 메타데이터 추출 실패: {}", e))?;
    if !quiet {
        eprintln!(
            "[auf-tool] Meta: arch={}, layers={}, vocab={}",
            meta.architecture, meta.n_layers, meta.vocab_size
        );
    }

    // 5) Tokenizer 구성
    let tokenizer = load_tokenizer_from_json(&args.tokenizer)?;

    // 6) ModelConfig (Q/K permute shape 결정용)
    let config = llm_rs2::models::config::ModelConfig::from_gguf_metadata(&gguf)
        .map_err(|e| anyhow!("ModelConfig 파싱 실패: {}", e))?;

    // 7) weight payload 생성 (각 variant 별)
    let mut writer = AufWriter::new(meta, tokenizer, source_hash, source_size, source_mtime);

    let created_by = args
        .created_by
        .unwrap_or_else(|| format!("llm_rs2 auf-tool v{}", env!("CARGO_PKG_VERSION")));
    writer = writer.with_created_by(&created_by);

    // Q4_0 tensor raw bytes 추출 (공통, variant마다 재사용)
    // 순서: layer 0..N의 wq, wk, wv, wo, w_gate, w_up, w_down 순
    // permute가 필요한 wq/wk에는 unpermute_qk_rows 적용
    // lm_head는 `lm_head_mode`에 따라 Q4_0 quantize 적용 (Sprint G-1).
    if !quiet {
        eprint!("[auf-tool] Extracting weight tensors...");
    }
    let (tensor_blobs, lm_head_q4_0_present) =
        extract_weight_blobs(&gguf, &config, lm_head_mode, quiet)?;
    if !quiet {
        eprintln!(" {} tensors extracted", tensor_blobs.len());
    }

    // capability_optional bit 2 설정 여부 결정 (Sprint G-1).
    // - lm_head Q4_0 entry가 실제로 추가/유지된 경우만 set.
    // - Off 모드 또는 tied 모델 (lm_head 없음)에서는 unset → v0.1.0 호환.
    writer = writer.with_lm_head_q4_0(lm_head_q4_0_present);
    if !quiet && lm_head_q4_0_present {
        eprintln!(
            "[auf-tool] LM_HEAD_PRECOMPUTED_Q4_0 capability bit 2 = 1 (format_patch=1, v0.1.1)"
        );
    }

    // TensorIndex 구성 (weights payload 추가 전에 offset 계산)
    let tensor_index = build_tensor_index(&tensor_blobs, &variant_tags);
    if !quiet {
        eprintln!(
            "[auf-tool] TensorIndex: {} variants, {} entries",
            tensor_index.variant_tags.len(),
            tensor_index.entries.len()
        );
    }
    writer = writer.with_tensor_index(tensor_index);

    for &tag in &variant_tags {
        if !quiet {
            eprint!("[auf-tool] Building {}...", tag);
        }
        let payload = build_variant_payload(&tensor_blobs, tag)?;
        let size = payload.len();
        writer = writer.add_weights_section(tag, payload);
        if !quiet {
            eprintln!(" {} MB", size / (1024 * 1024));
        }
    }

    // 8) Atomic write
    if !quiet {
        eprint!("[auf-tool] Writing {}...", args.output.display());
    }
    writer
        .write_to_file(&args.output)
        .map_err(|e| anyhow!("AUF 쓰기 실패: {}", e))?;

    if !quiet {
        let file_size = std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!(
            " done ({:.2} GiB)",
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        eprintln!("[auf-tool] Build complete: {}", args.output.display());
    }
    Ok(())
}

/// `(tensor_name, raw_bytes_after_permute, shape_logical)` 항목 타입.
///
/// `shape_logical`은 outermost-first (논리적 순서).
type WeightBlob = (String, Vec<u8>, Vec<u64>);

/// GGUF ggml_type 코드 (subset, lm_head 분기에서 사용).
///
/// `gguf.rs`의 private 상수와 동일하지만 binary crate에서 접근할 수 없으므로 로컬에 둔다.
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;

/// GGUF에서 lm_head를 식별하는 표준 이름 (untied / separate 모델).
const LM_HEAD_SEPARATE_NAME: &str = "output.weight";

/// GGUF에서 token embedding tensor 이름. tied embedding 모델 (Llama 3.2 1B/3B 등)에서
/// lm_head source로 재사용된다.
const LM_HEAD_TIED_SOURCE_NAME: &str = "token_embd.weight";

/// 선택된 lm_head source 정보 (`select_lm_head_source` 반환).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LmHeadSource {
    /// 실제 GGUF tensor 이름 (`output.weight` 또는 `token_embd.weight`).
    source_name: &'static str,
    /// `true`면 tied embedding 모델 (token_embd를 lm_head로 재사용).
    is_tied: bool,
}

/// GGUF에 존재하는 tensor를 기반으로 lm_head source를 선택한다.
///
/// 우선순위:
/// 1. `output.weight` 가 있으면 separate 모델 (Qwen, Llama 3 8B 등) → 그것을 사용.
/// 2. 없으면 `token_embd.weight` 를 tied source로 재사용 (Llama 3.2 1B/3B 등).
/// 3. 둘 다 없으면 `None` (entry 생성 불가).
///
/// 본 함수는 GgufFile 의존성 없이 boolean만 받아 결정하므로 단위 테스트가 용이하다.
fn select_lm_head_source(has_separate: bool, has_token_embd: bool) -> Option<LmHeadSource> {
    if has_separate {
        Some(LmHeadSource {
            source_name: LM_HEAD_SEPARATE_NAME,
            is_tied: false,
        })
    } else if has_token_embd {
        Some(LmHeadSource {
            source_name: LM_HEAD_TIED_SOURCE_NAME,
            is_tied: true,
        })
    } else {
        None
    }
}

/// `WeightBlob` 목록 추출 + lm_head Q4_0 사전 변환 (선택).
///
/// GGUF는 innermost-first로 dims를 저장하므로, 여기서 reverse하여 logical order로 반환한다.
/// AUF TensorEntry.shape에는 logical order(outermost-first)로 저장하며,
/// reader(`secondary_mmap.rs`)에서 다시 reverse하여 GGUF order로 복원한다.
///
/// `lm_head_mode`에 따라 lm_head를 다음과 같이 처리한다:
/// - `Off`: separate model에서 `output.weight`이 존재하면 raw bytes를 그대로 추출
///   (기존 동작). tied model에서는 lm_head entry 미생성 (v0.1.0 byte-level 호환).
/// - `On`/`Auto`: lm_head source 선택:
///   - separate model: `output.weight` 사용. dtype 분기:
///     - 이미 Q4_0이면 raw bytes 그대로 동봉.
///     - F16/F32 → F32 dequantize → Q4_0 quantize → 18B/block bytes로 교체.
///     - 그 외 dtype은 변환 미지원으로 raw bytes 동봉 + bit 미설정.
///   - tied model (Llama 3.2 1B 등): `token_embd.weight` 를 lm_head source로 재사용.
///     동일 dtype 분기 적용. tied 케이스에서도 lm_head Q4_0 entry는 별도 GPU 버퍼
///     (Sprint F의 `quantize_lm_head_to_q4_0()` 흐름)로 의미가 있다.
///
/// 반환 튜플 두 번째 값은 lm_head Q4_0 entry가 *실제로* 추가/유지되었는지 (capability bit 2 set
/// 결정에 사용). source가 둘 다 없거나 Off 모드면 false.
fn extract_weight_blobs(
    gguf: &llm_rs2::models::loader::gguf::GgufFile,
    config: &llm_rs2::models::config::ModelConfig,
    lm_head_mode: IncludeLmHeadMode,
    quiet: bool,
) -> Result<(Vec<WeightBlob>, bool)> {
    let mut blobs: Vec<WeightBlob> = Vec::new();
    let mut lm_head_q4_0_present = false;

    // cross-layer tensors. lm_head ("output.weight")은 모드에 따라 별도 처리.
    let regular_cross = ["token_embd.weight", "output_norm.weight"];
    for &name in &regular_cross {
        if let Some(info) = gguf.find_tensor(name) {
            let data = gguf.tensor_data(info);
            // GGUF dims는 innermost-first → reversed = outermost-first (logical)
            let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
            blobs.push((name.to_owned(), data.to_vec(), shape_logical));
        }
    }

    // lm_head — separate (`output.weight`) vs tied (`token_embd.weight` 재사용) 분기.
    //
    // Llama 3.2 1B 등 tied embedding 모델은 GGUF에 `output.weight`을 별도로 저장하지
    // 않고 `token_embd.weight`과 weight를 공유한다. Sprint F의
    // `quantize_lm_head_to_q4_0()`는 tied 모델에서도 별도 Q4_0 GPU 버퍼를 생성하여
    // matmul 효율을 얻으므로, AUF builder도 동일하게 tied source로부터 lm_head Q4_0
    // entry를 생성한다 (Sprint G-1-B fix, 2026-04-26).
    let lm_head_source = select_lm_head_source(
        gguf.find_tensor(LM_HEAD_SEPARATE_NAME).is_some(),
        gguf.find_tensor(LM_HEAD_TIED_SOURCE_NAME).is_some(),
    );

    if let Some(LmHeadSource {
        source_name,
        is_tied,
    }) = lm_head_source
    {
        if lm_head_mode == IncludeLmHeadMode::Off {
            // Off: separate 케이스에서만 raw bytes 동봉. tied에서는 entry 미생성
            // (token_embd는 이미 cross-layer로 추가됨).
            if !is_tied {
                let info = gguf.find_tensor(source_name).expect("source exists");
                let raw = gguf.tensor_data(info);
                let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
                blobs.push((
                    LM_HEAD_SEPARATE_NAME.to_owned(),
                    raw.to_vec(),
                    shape_logical,
                ));
            }
        } else {
            // On / Auto: source dtype 분기. tied/separate 무관하게 동일 처리.
            // 단, tied에서는 entry 이름은 항상 "output.weight" (lm_head 식별).
            let info = gguf
                .find_tensor(source_name)
                .expect("source exists per match arm above");
            let raw = gguf.tensor_data(info);

            // tied 케이스: entry shape은 token_embd shape과 동일 (== [vocab_size, hidden_dim]).
            // separate 케이스: source shape 그대로.
            let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
            let is_q4_0 = info.ggml_type == GGML_TYPE_Q4_0;
            let dtype_str = match info.ggml_type {
                GGML_TYPE_F32 => "F32",
                GGML_TYPE_F16 => "F16",
                GGML_TYPE_Q4_0 => "Q4_0",
                _ => "other",
            };
            let source_label = if is_tied {
                format!("{} (tied)", source_name)
            } else {
                source_name.to_owned()
            };

            // tied 안전망 — config.vocab_size / hidden_size와 일치 검증.
            // tied 모델에서는 token_embd.weight = lm_head이므로 shape이 [vocab, hidden]임이
            // 보장되어야 한다. 미일치 시 이후 quantize / GPU upload가 잘못된 cl_mem 크기를
            // 만들어 silent corruption을 유발하므로 fail-fast.
            if is_tied {
                let expected_rows = config.vocab_size as u64;
                let expected_cols = config.hidden_size as u64;
                if shape_logical.len() != 2
                    || shape_logical[0] != expected_rows
                    || shape_logical[1] != expected_cols
                {
                    bail!(
                        "tied lm_head source '{}' shape {:?} does not match config \
                         [vocab_size={}, hidden_dim={}]",
                        source_name,
                        shape_logical,
                        expected_rows,
                        expected_cols
                    );
                }
            }

            if is_q4_0 {
                // 이미 Q4_0 — quantize 불필요. AOS bytes를 그대로 동봉.
                if !quiet {
                    eprintln!(
                        "[auf-tool] lm_head: source={}, dtype={}, raw {} bytes, \
                         entry included as-is",
                        source_label,
                        dtype_str,
                        raw.len()
                    );
                }
                // tied: source는 token_embd지만 entry 이름은 lm_head 식별을 위해 output.weight.
                blobs.push((
                    LM_HEAD_SEPARATE_NAME.to_owned(),
                    raw.to_vec(),
                    shape_logical,
                ));
                lm_head_q4_0_present = true;
            } else if info.ggml_type == GGML_TYPE_F16 || info.ggml_type == GGML_TYPE_F32 {
                // F16/F32 → F32 dequantize → Q4_0 quantize.
                if !quiet {
                    eprintln!(
                        "[auf-tool] lm_head: source={}, dtype={}, quantizing to Q4_0...",
                        source_label, dtype_str,
                    );
                }
                // Sprint F와 일치: tied/separate 무관하게 token_embd / output.weight raw bytes를
                // F32 dequantize → quantize_q4_0. shape은 [rows=vocab, cols=hidden].
                if shape_logical.len() != 2 {
                    bail!(
                        "lm_head source '{}' must be 2-D (got shape={:?})",
                        source_name,
                        shape_logical
                    );
                }
                let q4_bytes =
                    quantize_lm_head_to_q4_0(raw, &shape_logical, info.ggml_type, quiet)?;
                blobs.push((LM_HEAD_SEPARATE_NAME.to_owned(), q4_bytes, shape_logical));
                lm_head_q4_0_present = true;
            } else {
                // 미지원 dtype (Q4_K, Q8_0 등) — fallback: raw bytes + bit 미설정.
                if !quiet {
                    eprintln!(
                        "[auf-tool] Warning: lm_head source={}, dtype={} (ggml_type={}) \
                         unsupported for Q4_0 quantize; entry included as raw bytes \
                         (capability bit 2 not set)",
                        source_label, dtype_str, info.ggml_type
                    );
                }
                // tied + 미지원 dtype 조합은 매우 드물고 (token_embd가 Q4_K 등) 엔진 path도
                // 처리하지 않으므로 entry를 만들지 않는다.
                if !is_tied {
                    blobs.push((
                        LM_HEAD_SEPARATE_NAME.to_owned(),
                        raw.to_vec(),
                        shape_logical,
                    ));
                }
            }
        }
    } else if lm_head_mode != IncludeLmHeadMode::Off && !quiet {
        // separate / tied 양쪽 모두 source가 없는 매우 드문 경우.
        eprintln!(
            "[auf-tool] Note: GGUF has neither 'output.weight' nor 'token_embd.weight'; \
             lm_head Q4_0 entry skipped, capability bit 2 not set."
        );
    }

    // per-layer tensors (모든 레이어)
    let n_layers = config.num_hidden_layers;
    let weight_kinds = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
    ];

    for layer in 0..n_layers {
        for &kind in &weight_kinds {
            let name = format!("blk.{layer}.{kind}");
            if let Some(info) = gguf.find_tensor(&name) {
                let raw = gguf.tensor_data(info);
                // GGUF dims는 innermost-first → reversed = outermost-first (logical)
                let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();

                // Q/K permute (Llama arch only, attn_q/attn_k)
                let bytes = if let Some((n_head, head_dim)) = qk_permute_shape_local(&name, config)
                {
                    let total_rows = n_head * head_dim;
                    let row_size = raw.len() / total_rows;
                    unpermute_qk_rows_local(raw, n_head, head_dim, row_size)
                } else {
                    raw.to_vec()
                };

                blobs.push((name, bytes, shape_logical));
            } else if !quiet && kind.ends_with(".weight") && !kind.contains("norm") {
                // weight tensor 누락 경고 (norm은 없을 수 있음)
                eprintln!("[auf-tool] Warning: tensor '{}' not found in GGUF", name);
            }
        }
    }

    Ok((blobs, lm_head_q4_0_present))
}

/// GGUF lm_head (F16 or F32) → F32 dequantize → Q4_0 quantize → 18B/block bytes.
///
/// `convert::quantize_q4_0`(layer weight도 동일하게 사용)을 재사용한다 — 신규 변환
/// 함수 작성 금지 (Sprint G-1-A 결정). 결정성은 `quantize_q4_0`의 단순 max-abs scaling
/// 루프가 부동소수점 reduction 순서를 고정하므로 호스트 결정적이다 (ENG-DAT-096.13).
///
/// 변환 시간을 stderr로 기록한다 (Sprint G-1 요구).
fn quantize_lm_head_to_q4_0(
    raw: &[u8],
    shape_logical: &[u64],
    ggml_type: u32,
    quiet: bool,
) -> Result<Vec<u8>> {
    use llm_rs2::core::quant::{BlockQ4_0, QK4_0};
    use llm_rs2::models::loader::convert::{f16_to_f32, quantize_q4_0};

    if shape_logical.len() != 2 {
        bail!("lm_head must be 2-D (got shape={:?})", shape_logical);
    }
    let rows = shape_logical[0] as usize; // outermost (vocab)
    let cols = shape_logical[1] as usize; // innermost (hidden)
    if !cols.is_multiple_of(QK4_0) {
        bail!(
            "lm_head inner dim ({}) is not a multiple of QK4_0 ({}); cannot quantize to Q4_0",
            cols,
            QK4_0
        );
    }
    let numel = rows * cols;

    let t0 = std::time::Instant::now();

    // Step 1: dequantize to F32 host buffer.
    let mut f32_data = vec![0.0f32; numel];
    match ggml_type {
        GGML_TYPE_F16 => {
            f16_to_f32(raw, &mut f32_data, numel);
        }
        GGML_TYPE_F32 => {
            // SAFETY: GGUF F32 is little-endian f32 array of length `numel`.
            let src = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, numel) };
            f32_data.copy_from_slice(src);
        }
        other => {
            bail!(
                "quantize_lm_head_to_q4_0: unsupported ggml_type={} (expected F16 or F32)",
                other
            );
        }
    }

    // Step 2: quantize to Q4_0 blocks (layer weight와 동일 함수).
    let blocks = quantize_q4_0(&f32_data, rows, cols);
    let block_bytes_each = std::mem::size_of::<BlockQ4_0>(); // 18B
    let total_bytes = blocks.len() * block_bytes_each;

    // SAFETY: BlockQ4_0 is `#[repr(C)]` with size 18 (asserted at compile time
    // in core::quant). Reading its bytes through `from_raw_parts` is well-defined.
    let block_bytes_view =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total_bytes) };
    let bytes_out = block_bytes_view.to_vec();

    let elapsed = t0.elapsed();
    if !quiet {
        eprintln!(
            "[auf-tool] lm_head Q4_0 quantize: {:.2}s, output {} MB ({}×{}, {} blocks)",
            elapsed.as_secs_f64(),
            total_bytes / (1024 * 1024),
            rows,
            cols,
            blocks.len(),
        );
    }
    Ok(bytes_out)
}

/// Llama Q/K weight 행 permute가 필요한지 판단하고 (n_head, head_dim) 반환.
///
/// `pub(crate)` 함수 `qk_permute_shape`을 바이너리에서 접근할 수 없으므로
/// 동등한 로직을 로컬에 구현한다.
fn qk_permute_shape_local(
    name: &str,
    config: &llm_rs2::models::config::ModelConfig,
) -> Option<(usize, usize)> {
    use llm_rs2::models::config::ModelArch;
    if config.arch != ModelArch::Llama {
        return None;
    }
    // "blk.<N>.attn_q.weight" 또는 "blk.<N>.attn_k.weight"
    let stem = name.strip_prefix("blk.")?;
    let (_idx, rest) = stem.split_once('.')?;
    let head_dim = config.head_dim;
    match rest {
        "attn_q.weight" => Some((config.num_attention_heads, head_dim)),
        "attn_k.weight" => Some((config.num_key_value_heads, head_dim)),
        _ => None,
    }
}

/// llama.cpp Q/K 행 permute를 되돌린다 (NeoX RoPE layout 복원).
///
/// `pub(crate)` 함수 `unpermute_qk_rows`를 바이너리에서 접근할 수 없으므로
/// 동등한 로직을 로컬에 구현한다.
fn unpermute_qk_rows_local(
    src: &[u8],
    n_head: usize,
    head_dim: usize,
    row_size_bytes: usize,
) -> Vec<u8> {
    let half = head_dim / 2;
    let total_rows = n_head * head_dim;
    debug_assert_eq!(src.len(), total_rows * row_size_bytes);
    let mut dst = vec![0u8; src.len()];
    for h in 0..n_head {
        let head_base = h * head_dim;
        for j in 0..head_dim {
            let src_in_head = if j < half { 2 * j } else { 2 * (j - half) + 1 };
            let src_row = head_base + src_in_head;
            let dst_row = head_base + j;
            let src_off = src_row * row_size_bytes;
            let dst_off = dst_row * row_size_bytes;
            dst[dst_off..dst_off + row_size_bytes]
                .copy_from_slice(&src[src_off..src_off + row_size_bytes]);
        }
    }
    dst
}

/// 특정 variant tag에 맞는 payload 바이트열 생성.
fn build_variant_payload(blobs: &[WeightBlob], tag: &str) -> Result<Vec<u8>> {
    match tag {
        TAG_WEIGHTS_ADRENO_SOA => {
            // SOA: layer Q4_0 weight를 (q_buf, d_buf)로 변환 + transpose하여 연속 배치.
            // 비-Q4_0 tensor (F16 norm 등)는 그대로 포함.
            //
            // **lm_head 예외 (G-1-F fix, INV-135 v2)**: lm_head의 q_buf size는 vocab×hidden
            // 차원 (Llama 3.2 1B: 32M texels)으로 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE`
            // 한계를 거의 모든 디바이스에서 초과한다. 따라서 `image1d_buffer_t` 생성이
            // 실패하여 빠른 SOA GEMV path를 발동시킬 수 없다. SOA 변환을 적용하면 reader
            // 측이 SOA layout을 가정하지만 forward는 standard GEMV(AOS layout 가정)로
            // 떨어져 silent corruption이 발생한다 (Sprint G-1-F 디바이스 측정에서 확인).
            //
            // 그러므로 lm_head Q4_0 entry는 ADRENO_SOA section 내부에서도 AOS 18B/block
            // layout으로 동봉한다. reader는 `entry.kind == LmHead` 식별로 AOS path를
            // 사용한다 (transformer.rs::load_lm_head_from_auf 참조).
            //
            // Q4_0 weight의 logical shape는 outermost-first `[ne01, ne00]` =
            // `[rows, cols]`로 저장되어 있다 (`extract_weight_blobs` 참조).
            // `q4_0_aos_to_adreno_soa`는 이 shape를 받아 `convert_q4_0_to_noshuffle`
            // 와 동등한 (a) nibble unshuffle (b) ushort q transpose (c) half d
            // transpose 를 빌드 타임에 적용한다. 결과 byte sequence는 backend의
            // `alloc_pre_converted_soa_tensor`가 직접 cl_mem에 업로드 가능한
            // 형태이다.
            let mut out = Vec::new();
            for (name, bytes, shape) in blobs {
                let is_lm_head = name == LM_HEAD_SEPARATE_NAME;
                let is_q4_0 = bytes.len() % 18 == 0 && bytes.len() >= 18;
                if !is_lm_head && is_q4_0 && shape.len() == 2 {
                    let ne01 = shape[0] as usize; // rows
                    let ne00 = shape[1] as usize; // cols (K dim)
                    // Defensive guard — if shape × 18B/block does not match the
                    // tensor byte count, fall back to byte-as-is so we do not
                    // corrupt the payload. This branch should not trigger for
                    // well-formed GGUF inputs.
                    let expected = (ne01 * ne00 / 32) * 18;
                    if expected == bytes.len() && ne00.is_multiple_of(32) {
                        let (q_buf, d_buf) = q4_0_aos_to_adreno_soa(bytes, ne00, ne01);
                        out.extend_from_slice(&q_buf);
                        out.extend_from_slice(&d_buf);
                        continue;
                    }
                    eprintln!(
                        "[auf-tool] Warning: SOA shape guard rejected '{}' (shape={:?}, bytes={}); \
                         emitting AOS bytes (forward path will fall back to AOS GEMV).",
                        name,
                        shape,
                        bytes.len()
                    );
                }
                // F16/F32 tensor, lm_head Q4_0 (image-limit exception),
                // 또는 shape-mismatch fallback: bytes as-is.
                out.extend_from_slice(bytes);
            }
            Ok(out)
        }
        TAG_WEIGHTS_CUDA_AOS => {
            // AOS + 128B align
            let mut out = Vec::new();
            for (_name, bytes, _shape) in blobs {
                let padded = q4_0_aos_with_align(bytes, 128);
                out.extend_from_slice(&padded);
            }
            Ok(out)
        }
        TAG_WEIGHTS_CPU_AOS => {
            // AOS + 64B align (NEON dotprod)
            let mut out = Vec::new();
            for (_name, bytes, _shape) in blobs {
                let padded = q4_0_aos_with_align(bytes, 64);
                out.extend_from_slice(&padded);
            }
            Ok(out)
        }
        _ => bail!("Unknown variant tag: {}", tag),
    }
}

// ---------------------------------------------------------------------------
// TensorIndex 구성 헬퍼
// ---------------------------------------------------------------------------

/// blob 이름 → `(layer_idx, TensorKind)` 변환.
///
/// cross-layer tensor는 `LAYER_IDX_CROSS` 반환.
/// 인식할 수 없는 이름은 `None` 반환.
fn tensor_name_to_layer_kind(name: &str) -> Option<(u32, TensorKind)> {
    // cross-layer tensors
    match name {
        "token_embd.weight" => return Some((LAYER_IDX_CROSS, TensorKind::Embedding)),
        "output_norm.weight" => return Some((LAYER_IDX_CROSS, TensorKind::FinalNorm)),
        "output.weight" => return Some((LAYER_IDX_CROSS, TensorKind::LmHead)),
        _ => {}
    }

    // per-layer: "blk.<N>.<kind>"
    let rest = name.strip_prefix("blk.")?;
    let (idx_str, kind_str) = rest.split_once('.')?;
    let layer_idx: u32 = idx_str.parse().ok()?;

    let kind = match kind_str {
        "attn_q.weight" => TensorKind::AttnQ,
        "attn_k.weight" => TensorKind::AttnK,
        "attn_v.weight" => TensorKind::AttnV,
        "attn_output.weight" => TensorKind::AttnO,
        "ffn_gate.weight" => TensorKind::FfnGate,
        "ffn_up.weight" => TensorKind::FfnUp,
        "ffn_down.weight" => TensorKind::FfnDown,
        "attn_norm.weight" => TensorKind::AttnNorm,
        "ffn_norm.weight" => TensorKind::FfnNorm,
        _ => return None,
    };

    Some((layer_idx, kind))
}

/// blob bytes가 특정 variant에서 차지하는 payload 크기를 계산한다.
///
/// `build_variant_payload`와 동일한 변환 로직을 적용하되, 실제 데이터 복사 없이
/// 크기만 반환한다.
fn compute_variant_tensor_size(bytes: &[u8], tag: &str) -> usize {
    match tag {
        TAG_WEIGHTS_ADRENO_SOA => {
            // Q4_0 SOA: q_buf(N*16) + d_buf(N*2) = N*18 = bytes.len() (불변)
            // 비-Q4_0: 그대로
            bytes.len()
        }
        TAG_WEIGHTS_CUDA_AOS => {
            // 128B align-up
            bytes.len().div_ceil(128) * 128
        }
        TAG_WEIGHTS_CPU_AOS => {
            // 64B align-up
            bytes.len().div_ceil(64) * 64
        }
        _ => bytes.len(),
    }
}

/// blob bytes의 dtype을 추정한다.
///
/// Q4_0: 18B 배수 → `TensorDType::Q4_0`
/// F16: 2B 배수 (기타) → `TensorDType::F16`
/// 그 외: `TensorDType::F32`
fn infer_dtype(bytes: &[u8]) -> TensorDType {
    if bytes.len() >= 18 && bytes.len().is_multiple_of(18) {
        TensorDType::Q4_0
    } else if bytes.len().is_multiple_of(2) {
        TensorDType::F16
    } else {
        TensorDType::F32
    }
}

/// `extract_weight_blobs` 결과와 variant tag 목록으로 `TensorIndex`를 구성한다.
///
/// 각 variant마다 payload 내 tensor별 section-local offset을 추적하여
/// `TensorEntry::variant_offsets`/`variant_sizes`를 채운다.
/// `TensorEntry::shape`는 logical order (outermost-first)로 채운다.
fn build_tensor_index(blobs: &[WeightBlob], variant_tags: &[&str]) -> TensorIndex {
    let variant_count = variant_tags.len();

    // variant_tags → [u8; 24] 배열 변환
    let vt_bytes: Vec<[u8; 24]> = variant_tags
        .iter()
        .map(|tag| {
            let mut buf = [0u8; 24];
            let b = tag.as_bytes();
            buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
            buf
        })
        .collect();

    // variant별 현재 누적 offset (section-local, 0-based)
    let mut variant_cursors: Vec<u64> = vec![0u64; variant_count];

    let mut entries: Vec<TensorEntry> = Vec::with_capacity(blobs.len());

    for (name, bytes, shape_logical) in blobs {
        let Some((layer_idx, kind)) = tensor_name_to_layer_kind(name) else {
            // 인식 불가 tensor: offset만 전진, entry 미등록
            for (vi, &tag) in variant_tags.iter().enumerate() {
                let sz = compute_variant_tensor_size(bytes, tag) as u64;
                variant_cursors[vi] += sz;
            }
            continue;
        };

        let dtype = infer_dtype(bytes);

        let mut variant_offsets = Vec::with_capacity(variant_count);
        let mut variant_sizes = Vec::with_capacity(variant_count);

        for (vi, &tag) in variant_tags.iter().enumerate() {
            let sz = compute_variant_tensor_size(bytes, tag) as u64;
            variant_offsets.push(variant_cursors[vi]);
            variant_sizes.push(sz);
            variant_cursors[vi] += sz;
        }

        entries.push(TensorEntry {
            layer_idx,
            kind: kind.as_u32(),
            dtype: dtype.as_u32(),
            // logical order (outermost-first): reader가 .rev()하여 GGUF innermost-first로 복원
            shape: shape_logical.clone(),
            alignment: 64,
            variant_offsets,
            variant_sizes,
        });
    }

    TensorIndex {
        variant_tags: vt_bytes,
        entries,
    }
}

// ---------------------------------------------------------------------------
// info 커맨드
// ---------------------------------------------------------------------------

fn cmd_info(args: InfoArgs) -> Result<()> {
    let path = &args.file;
    let view = open(path, BackendTag::Any)
        .map_err(|e| anyhow!("AUF 파일 열기 실패 ({}): {}", path.display(), e))?;

    let file_size = view.file_size();
    println!("File: {}", path.display());
    println!(
        "Size: {} bytes ({:.2} GiB)",
        file_size,
        file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    // Header
    let h = &view.header;
    let created_by = {
        let end = h.created_by.iter().position(|&b| b == 0).unwrap_or(32);
        String::from_utf8_lossy(&h.created_by[..end]).into_owned()
    };
    let source_hash_hex: String = h.source_hash.iter().map(|b| format!("{:02x}", b)).collect();
    let source_mtime_str = format_unix_time(h.source_mtime);

    println!("Header:");
    println!("  magic            : \"ARGUS_W\\0\"");
    println!(
        "  format           : v{}.{}.{}{}",
        h.format_major,
        h.format_minor,
        h.format_patch,
        if h.format_major == 0 {
            " (experimental)"
        } else {
            ""
        }
    );
    println!("  created_by       : \"{}\"", created_by);
    println!("  source_hash      : {}...", &source_hash_hex[..16]);
    println!("  source_size      : {} bytes", h.source_size);
    println!("  source_mtime     : {}", source_mtime_str);
    println!("  capability_req   : {:#018x}", h.capability_required);
    println!(
        "  capability_opt   : {:#018x}{}",
        h.capability_optional,
        if h.has_lm_head_q4_0() {
            " (LM_HEAD_PRECOMPUTED_Q4_0)"
        } else {
            ""
        }
    );
    println!("  section_count    : {}", h.section_count);
    println!(
        "  payload_start    : {:#x} ({})",
        h.payload_start_offset, h.payload_start_offset
    );
    println!(
        "  section_table_at : {:#x} ({})",
        h.section_table_offset, h.section_table_offset
    );
    println!();

    // Section table
    println!(
        "{:<26} {:<12} {:<16} {:<12} Version",
        "Tag", "Offset", "Size", "Flags"
    );
    println!("{}", "-".repeat(76));
    let mut variants_present = Vec::new();
    for entry in &view.section_table.entries {
        let tag = entry.tag();
        let flags_str = build_flags_str(entry.flags);
        let size_str = if entry.size >= 1024 * 1024 {
            format!("{:.1} MB", entry.size as f64 / (1024.0 * 1024.0))
        } else if entry.size >= 1024 {
            format!("{:.1} KB", entry.size as f64 / 1024.0)
        } else {
            format!("{} B", entry.size)
        };
        println!(
            "{:<26} {:#010x}   {:<16} {:<12} {}",
            tag, entry.offset, size_str, flags_str, entry.version
        );

        if tag.starts_with("WEIGHTS_") {
            let variant_name = tag.strip_prefix("WEIGHTS_").unwrap_or(tag);
            variants_present.push(variant_name.to_owned());
        }
    }
    println!();

    // META summary
    let m = &view.meta;
    println!("META:");
    println!("  architecture     : {}", m.architecture);
    println!("  n_layers         : {}", m.n_layers);
    println!("  n_heads_q        : {}", m.n_heads_q);
    println!("  n_kv_heads       : {}", m.n_kv_heads);
    println!("  head_dim         : {}", m.head_dim);
    println!("  hidden_dim       : {}", m.hidden_dim);
    println!("  ffn_dim          : {}", m.ffn_dim);
    println!("  vocab_size       : {}", m.vocab_size);
    println!("  max_seq_len      : {}", m.max_seq_len);
    println!("  rope_theta       : {}", m.rope_theta);
    println!("  rms_norm_epsilon : {}", m.rms_norm_epsilon);
    println!();

    // Tokenizer summary
    let t = &view.tokenizer;
    println!("TOKENIZER:");
    println!("  kind             : {} (BPE)", t.kind);
    println!("  vocab_size       : {}", t.tokens.len());
    println!("  merges_count     : {}", t.merges.len());
    println!(
        "  bos_id           : {}",
        if t.bos_id < 0 {
            "N/A".to_owned()
        } else {
            t.bos_id.to_string()
        }
    );
    println!(
        "  eos_id           : {}",
        if t.eos_id < 0 {
            "N/A".to_owned()
        } else {
            t.eos_id.to_string()
        }
    );
    println!(
        "  chat_template    : {}",
        if t.chat_template.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    println!();

    // TENSOR_INDEX summary
    let ti = &view.tensor_index;
    println!("TENSOR_INDEX:");
    println!("  variants         : {:?}", ti.variant_tag_strings());
    println!("  tensor_count     : {}", ti.entries.len());
    println!();

    // Variants present
    if variants_present.is_empty() {
        println!("Variants present: (none)");
    } else {
        println!("Variants present: {}", variants_present.join(", "));
    }

    Ok(())
}

fn build_flags_str(flags: u32) -> String {
    use llm_rs2::auf::{SECTION_COMPRESSED, SECTION_REQUIRED, SECTION_STRIPPABLE};
    let mut parts = Vec::new();
    if flags & SECTION_REQUIRED != 0 {
        parts.push("REQUIRED");
    }
    if flags & SECTION_STRIPPABLE != 0 {
        parts.push("STRIPPABLE");
    }
    if flags & SECTION_COMPRESSED != 0 {
        parts.push("COMPRESSED");
    }
    if parts.is_empty() {
        "0".to_owned()
    } else {
        parts.join("+")
    }
}

fn format_unix_time(secs: u64) -> String {
    if secs == 0 {
        return "N/A".to_owned();
    }
    // 간단한 UTC 포맷 (외부 crate 없이)
    // secs → 년월일시분초 계산
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // 년월일 계산 (간략, 윤년 근사)
    let mut year = 1970u64;
    let mut days = days_since_epoch;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months = [
        31u64,
        if is_leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 0usize;
    for &m in &months {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year,
        month + 1,
        days + 1,
        hours,
        minutes,
        seconds
    )
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

// ---------------------------------------------------------------------------
// strip 커맨드
// ---------------------------------------------------------------------------

fn cmd_strip(args: StripArgs) -> Result<()> {
    let path = &args.file;

    // keep tags 파싱
    let keep_tags_raw: Vec<String> = args
        .keep
        .split(',')
        .map(|s| normalize_tag(s.trim()))
        .collect();

    let keep_tags_str: Vec<&str> = keep_tags_raw.iter().map(|s| s.as_str()).collect();

    eprintln!("[auf-tool] Strip: {}", path.display());
    eprintln!("[auf-tool] Keep tags: {:?}", keep_tags_str);
    if !args.no_backup {
        let bak = path.with_extension("auf.bak");
        eprintln!("[auf-tool] Backup: {}", bak.display());
    }

    strip(path, &keep_tags_str, args.no_backup).map_err(|e| anyhow!("Strip 실패: {}", e))?;

    let new_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[auf-tool] Strip complete: {} ({:.2} GiB)",
        path.display(),
        new_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

/// 사용자 입력 variant tag를 정규화 (대소문자 무관, 축약형 허용).
fn normalize_tag(s: &str) -> String {
    match s.to_ascii_uppercase().as_str() {
        "ADRENO_SOA" | "ADRENO" => TAG_WEIGHTS_ADRENO_SOA.to_owned(),
        "CUDA_AOS" | "CUDA" => TAG_WEIGHTS_CUDA_AOS.to_owned(),
        "CPU_AOS" | "CPU" => TAG_WEIGHTS_CPU_AOS.to_owned(),
        // 이미 전체 tag 형식이거나 required section tag
        other => other.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// verify 커맨드
// ---------------------------------------------------------------------------

fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let path = &args.file;
    let mut all_pass = true;

    println!("Verifying: {}", path.display());
    println!("{}", "=".repeat(60));

    // 파일 읽기 (BackendTag::Any — backend 검증 우회)
    let view_result = open(path, BackendTag::Any);

    match &view_result {
        Ok(_) => {}
        Err(AufError::FileTooSmall) => {
            print_check("File size >= 256B", false, "파일이 너무 작습니다 (< 256B)");
            all_pass = false;
            print_result(all_pass);
            return if all_pass {
                Ok(())
            } else {
                Err(anyhow!("Verify FAILED"))
            };
        }
        Err(e) => {
            // 특정 에러를 검증 항목별로 분류
            let check_name = classify_error(e);
            print_check(check_name, false, &e.to_string());
            all_pass = false;
            print_result(all_pass);
            return Err(anyhow!("Verify FAILED"));
        }
    }

    let view = view_result.unwrap();

    // INV-132: magic, format_major, capability_required
    print_check("Magic bytes (ARGUS_W\\0)", true, "");
    print_check("format_major <= READER_MAX", true, "");
    print_check("capability_required all known bits", true, "");

    // INV-133: required sections
    let required_sections = ["META", "TOKENIZER", "TENSOR_INDEX"];
    for &tag in &required_sections {
        let found = view.section_table.find(tag).is_some();
        print_check(
            &format!("Required section: {}", tag),
            found,
            if found { "" } else { "누락" },
        );
        if !found {
            all_pass = false;
        }
    }

    // INV-134: section table 무결성 (open()에서 이미 검증됨)
    print_check("Section tags unique", true, "");
    print_check("Section flags valid (no REQUIRED+STRIPPABLE)", true, "");
    print_check("No COMPRESSED sections (v0.1)", true, "");
    print_check("Section offset/size within file bounds", true, "");
    print_check("Section ranges no overlap", true, "");

    // META 파싱 확인
    let meta_ok = !view.meta.architecture.is_empty() && view.meta.n_layers > 0;
    print_check(
        "META JSON parseable",
        meta_ok,
        if meta_ok {
            ""
        } else {
            "파싱 실패 또는 필드 비어 있음"
        },
    );
    if !meta_ok {
        all_pass = false;
    }

    // TOKENIZER 파싱 확인
    let tok_ok = !view.tokenizer.tokens.is_empty();
    print_check(
        "TOKENIZER vocab non-empty",
        tok_ok,
        if tok_ok { "" } else { "vocab이 비어 있음" },
    );
    if !tok_ok {
        all_pass = false;
    }

    // Variant sections 정보 출력 (pass/fail 아닌 informational)
    let variant_tags = [
        TAG_WEIGHTS_ADRENO_SOA,
        TAG_WEIGHTS_CUDA_AOS,
        TAG_WEIGHTS_CPU_AOS,
    ];
    let mut found_variants = Vec::new();
    for &tag in &variant_tags {
        if let Some(entry) = view.section_table.find(tag) {
            let strippable = entry.flags & SECTION_STRIPPABLE != 0;
            print_check(
                &format!("Variant section: {}", tag),
                strippable,
                if strippable {
                    "present, STRIPPABLE"
                } else {
                    "present, flag mismatch"
                },
            );
            found_variants.push(tag);
        }
    }
    if found_variants.is_empty() {
        println!("  [INFO] No WEIGHTS_* variant sections present");
    }

    // TENSOR_INDEX 내용 검증: WEIGHTS_* section이 존재하면 TENSOR_INDEX가 cover해야 함
    if !found_variants.is_empty() {
        let ti = &view.tensor_index;
        for &weights_tag in &found_variants {
            let covered = ti.variant_index_for_tag(weights_tag).is_some();
            let check_name = format!("TENSOR_INDEX covers variant: {weights_tag}");
            if covered {
                print_check(&check_name, true, "");
            } else {
                print_check(
                    &check_name,
                    false,
                    &format!(
                        "TENSOR_INDEX does not cover variant '{weights_tag}' — \
                         empty index detected. Rebuild with 'auf-tool build'."
                    ),
                );
                all_pass = false;
            }
        }

        // tensor_count가 0이면 추가 경고
        if ti.entries.is_empty() && !found_variants.is_empty() {
            print_check(
                "TENSOR_INDEX has entries (tensor_count > 0)",
                false,
                "TENSOR_INDEX has 0 entries but WEIGHTS_* sections exist — \
                 rebuild required",
            );
            all_pass = false;
        } else if !ti.entries.is_empty() {
            print_check(
                &format!("TENSOR_INDEX tensor_count={}", ti.entries.len()),
                true,
                "",
            );
        }

        // 각 TensorEntry.shape rank > 0 검증 (빈 shape = swap_executor 차단)
        let mut all_shapes_ok = true;
        for entry in &ti.entries {
            if entry.shape.is_empty() {
                let layer_str = if entry.layer_idx == u32::MAX {
                    "cross".to_owned()
                } else {
                    entry.layer_idx.to_string()
                };
                print_check(
                    &format!(
                        "TensorEntry shape non-empty (layer={} kind={})",
                        layer_str, entry.kind
                    ),
                    false,
                    &format!(
                        "entry layer={} kind={} has empty shape — \
                         TensorEntry.shape must be populated. Rebuild with 'auf-tool build'.",
                        layer_str, entry.kind
                    ),
                );
                all_shapes_ok = false;
                all_pass = false;
            }
        }
        if all_shapes_ok && !ti.entries.is_empty() {
            print_check("All TensorEntry shapes non-empty", true, "");
        }
    }

    // source_hash 비교 (--source 옵션)
    if let Some(source_path) = &args.source {
        eprint!("  Computing source_hash for {}...", source_path.display());
        match compute_source_hash(source_path) {
            Ok((computed_hash, _, _)) => {
                eprintln!(" done");
                let matches = computed_hash == view.header.source_hash;
                if matches {
                    print_check("source_hash matches GGUF", true, "");
                } else {
                    let auf_hash: String = view
                        .header
                        .source_hash
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect();
                    let comp_hash: String =
                        computed_hash.iter().map(|b| format!("{:02x}", b)).collect();
                    println!("  [WARN] source_hash mismatch (informational, not a hard failure)");
                    println!("         AUF says : {}...", &auf_hash[..16]);
                    println!("         Computed : {}...", &comp_hash[..16]);
                }
            }
            Err(e) => {
                eprintln!();
                println!("  [WARN] source_hash 계산 실패: {}", e);
            }
        }
    }

    println!("{}", "-".repeat(60));
    print_result(all_pass);

    if all_pass {
        Ok(())
    } else {
        Err(anyhow!("Verify FAILED"))
    }
}

fn print_check(name: &str, pass: bool, detail: &str) {
    let mark = if pass { "PASS" } else { "FAIL" };
    if detail.is_empty() {
        println!("  [{mark}] {name}");
    } else {
        println!("  [{mark}] {name}: {detail}");
    }
}

fn print_result(all_pass: bool) {
    if all_pass {
        println!("Result: PASS — AUF is valid");
    } else {
        println!("Result: FAIL — AUF has integrity issues");
    }
}

fn classify_error(e: &AufError) -> &'static str {
    match e {
        AufError::MagicMismatch => "Magic bytes (ARGUS_W\\0)",
        AufError::UnsupportedFormatMajor { .. } => "format_major <= READER_MAX",
        AufError::UnknownRequiredCapability { .. } => "capability_required all known bits",
        AufError::RequiredSectionMissing { .. } => "Required section present",
        AufError::SectionTableTruncated => "Section table within file bounds",
        AufError::SectionRangeInvalid { .. } => "Section offset/size within file bounds",
        AufError::SectionOverlap { .. } => "Section ranges no overlap",
        AufError::DuplicateSectionTag { .. } => "Section tags unique",
        AufError::ContradictoryFlags { .. } => "Section flags valid",
        AufError::CompressedSectionUnsupported { .. } => "No COMPRESSED sections",
        _ => "AUF parsing",
    }
}

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_variants_all() {
        let tags = parse_variants("all").unwrap();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&TAG_WEIGHTS_ADRENO_SOA));
        assert!(tags.contains(&TAG_WEIGHTS_CUDA_AOS));
        assert!(tags.contains(&TAG_WEIGHTS_CPU_AOS));
    }

    #[test]
    fn parse_variants_comma() {
        let tags = parse_variants("adreno_soa,cpu_aos").unwrap();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&TAG_WEIGHTS_ADRENO_SOA));
        assert!(tags.contains(&TAG_WEIGHTS_CPU_AOS));
        assert!(!tags.contains(&TAG_WEIGHTS_CUDA_AOS));
    }

    #[test]
    fn parse_variants_full_tag_name() {
        let tags = parse_variants("WEIGHTS_CUDA_AOS").unwrap();
        assert_eq!(tags.len(), 1);
        assert!(tags.contains(&TAG_WEIGHTS_CUDA_AOS));
    }

    #[test]
    fn parse_variants_unknown_fails() {
        let result = parse_variants("unknown_backend");
        assert!(result.is_err());
    }

    #[test]
    fn parse_variants_dedup() {
        let tags = parse_variants("adreno_soa,adreno_soa,cpu_aos").unwrap();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn normalize_tag_adreno() {
        assert_eq!(normalize_tag("adreno_soa"), TAG_WEIGHTS_ADRENO_SOA);
        assert_eq!(normalize_tag("ADRENO"), TAG_WEIGHTS_ADRENO_SOA);
        assert_eq!(normalize_tag("WEIGHTS_ADRENO_SOA"), TAG_WEIGHTS_ADRENO_SOA);
    }

    #[test]
    fn normalize_tag_pass_through() {
        assert_eq!(normalize_tag("META"), "META");
        assert_eq!(normalize_tag("TOKENIZER"), "TOKENIZER");
    }

    #[test]
    fn q4_0_aos_to_soa_basic() {
        // 1 block: 2B scale + 16B nibbles
        let mut block = vec![0u8; 18];
        block[0] = 0xAB;
        block[1] = 0xCD; // scale f16
        for (i, item) in block[2..18].iter_mut().enumerate() {
            *item = (i + 2) as u8; // nibbles
        }
        let (q_buf, d_buf) = q4_0_aos_to_soa(&block);
        assert_eq!(q_buf.len(), 16);
        assert_eq!(d_buf.len(), 2);
        assert_eq!(d_buf[0], 0xAB);
        assert_eq!(d_buf[1], 0xCD);
        assert_eq!(q_buf[0], 2);
        assert_eq!(q_buf[15], 17);
    }

    #[test]
    fn q4_0_aos_to_soa_multi_block() {
        let n = 4;
        let block = vec![0xFFu8; 18 * n]; // n blocks
        let (q_buf, d_buf) = q4_0_aos_to_soa(&block);
        assert_eq!(q_buf.len(), 16 * n);
        assert_eq!(d_buf.len(), 2 * n);
    }

    #[test]
    fn q4_0_aos_with_align_cpu() {
        let data = vec![1u8; 100];
        let out = q4_0_aos_with_align(&data, 64);
        assert_eq!(out.len(), 128); // ceil(100/64)*64
        assert_eq!(&out[..100], data.as_slice());
        assert_eq!(&out[100..], vec![0u8; 28].as_slice());
    }

    #[test]
    fn q4_0_aos_with_align_exact() {
        let data = vec![1u8; 64];
        let out = q4_0_aos_with_align(&data, 64);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn format_unix_time_epoch() {
        let s = format_unix_time(0);
        assert_eq!(s, "N/A");
    }

    #[test]
    fn format_unix_time_known() {
        // 2026-04-25 00:00:00 UTC = 1777075200
        let s = format_unix_time(1777075200);
        assert!(s.contains("2026"), "Expected 2026 in '{}'", s);
        assert!(s.contains("04"), "Expected month 04 in '{}'", s);
        assert!(s.contains("25"), "Expected day 25 in '{}'", s);
    }

    #[test]
    fn parse_include_lm_head_modes() {
        assert_eq!(
            IncludeLmHeadMode::parse("auto").unwrap(),
            IncludeLmHeadMode::Auto
        );
        assert_eq!(
            IncludeLmHeadMode::parse("AUTO").unwrap(),
            IncludeLmHeadMode::Auto
        );
        assert_eq!(
            IncludeLmHeadMode::parse("on").unwrap(),
            IncludeLmHeadMode::On
        );
        assert_eq!(
            IncludeLmHeadMode::parse("off").unwrap(),
            IncludeLmHeadMode::Off
        );
        assert_eq!(
            IncludeLmHeadMode::parse(" off ").unwrap(),
            IncludeLmHeadMode::Off
        );
        assert!(IncludeLmHeadMode::parse("invalid").is_err());
    }

    #[test]
    fn lm_head_q4_0_quantize_deterministic() {
        // 같은 F32 입력으로 두 번 quantize 했을 때 byte-level 동일한지 확인.
        // ENG-DAT-096.13 (writer 결정성) 핵심 invariant.
        let rows = 4usize;
        let cols = 64usize; // 2 blocks/row × 4 rows = 8 blocks → 144 bytes
        let mut f32_buf = vec![0.0f32; rows * cols];
        // pseudo-random 시퀀스 (deterministic seed).
        for (i, v) in f32_buf.iter_mut().enumerate() {
            *v = ((i as f32) * 0.0173).sin();
        }
        // F32 → little-endian byte slice.
        let raw_bytes: Vec<u8> = f32_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out1 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F32, true).unwrap();
        let out2 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F32, true).unwrap();
        assert_eq!(out1, out2, "lm_head Q4_0 quantize must be deterministic");
        // 8 blocks × 18B = 144B
        assert_eq!(out1.len(), rows * cols / 32 * 18);
    }

    #[test]
    fn lm_head_q4_0_quantize_rejects_non_q4_0_aligned_cols() {
        let shape: Vec<u64> = vec![4, 17]; // 17 is not a multiple of 32
        let raw = vec![0u8; 4 * 17 * 4];
        let err = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_F32, true).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("multiple of QK4_0"), "got: {msg}");
    }

    #[test]
    fn lm_head_q4_0_quantize_rejects_unsupported_dtype() {
        let shape: Vec<u64> = vec![1, 32];
        let raw = vec![0u8; 32 * 4];
        // GGML_TYPE_Q4_0 (already quantized) should NOT be re-quantized via this helper.
        let err = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_Q4_0, true).unwrap_err();
        assert!(format!("{err}").contains("unsupported ggml_type"));
    }

    #[test]
    fn writer_with_lm_head_q4_0_sets_bit2_and_patch() {
        use llm_rs2::auf::header::CAPABILITY_BIT_LM_HEAD_Q4_0;
        use llm_rs2::auf::{AufHeader, AufMeta, AufTokenizer, AufWriter};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };

        // (1) with_lm_head_q4_0(true) → bit 2 set, format_patch = 1.
        let bytes_on = AufWriter::new(meta.clone(), tok.clone(), [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 64])
            .with_lm_head_q4_0(true)
            .build()
            .unwrap();
        let hdr_on = AufHeader::from_bytes(&bytes_on).unwrap();
        assert_eq!(hdr_on.format_patch, 1);
        assert_ne!(
            hdr_on.capability_optional & CAPABILITY_BIT_LM_HEAD_Q4_0,
            0,
            "bit 2 must be set"
        );
        assert!(hdr_on.has_lm_head_q4_0());

        // (2) with_lm_head_q4_0(false) (default) → bit 2 clear, format_patch = 0.
        let bytes_off = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 64])
            .build()
            .unwrap();
        let hdr_off = AufHeader::from_bytes(&bytes_off).unwrap();
        assert_eq!(hdr_off.format_patch, 0);
        assert_eq!(hdr_off.capability_optional & CAPABILITY_BIT_LM_HEAD_Q4_0, 0);
        assert!(!hdr_off.has_lm_head_q4_0());
    }

    #[test]
    fn writer_off_byte_level_compatible_with_v0_1_0() {
        // INV-136 회귀 방지: Off 모드 (capability bit 2 = 0)에서 작성된 AUF 헤더는
        // v0.1.0 출력과 동일한 format 필드를 가져야 한다 (format_patch=0, capability=0).
        use llm_rs2::auf::{AufHeader, AufMeta, AufTokenizer, AufWriter};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 0,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };
        // 명시적으로 false 호출 vs 기본값 모두 동일 결과여야 함.
        let bytes_default = AufWriter::new(meta.clone(), tok.clone(), [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![0xAAu8; 64])
            .build()
            .unwrap();
        let bytes_explicit_off = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![0xAAu8; 64])
            .with_lm_head_q4_0(false)
            .build()
            .unwrap();
        assert_eq!(
            bytes_default, bytes_explicit_off,
            "with_lm_head_q4_0(false) must match default"
        );
        let hdr = AufHeader::from_bytes(&bytes_default).unwrap();
        assert_eq!(hdr.format_major, 0);
        assert_eq!(hdr.format_minor, 1);
        assert_eq!(hdr.format_patch, 0);
        assert_eq!(hdr.capability_required, 0);
        assert_eq!(hdr.capability_optional, 0);
    }

    // ── Sprint G-1-B fix: tied embedding lm_head source ─────────────────────

    #[test]
    fn tied_model_uses_token_embd_as_lm_head_source() {
        // Llama 3.2 1B: GGUF에 output.weight 없음 (tied) → token_embd.weight를
        // lm_head source로 선택해야 한다. is_tied = true.
        let src = select_lm_head_source(false, true).expect("tied source must be selected");
        assert_eq!(src.source_name, LM_HEAD_TIED_SOURCE_NAME);
        assert_eq!(src.source_name, "token_embd.weight");
        assert!(src.is_tied);
    }

    #[test]
    fn separate_model_uses_output_weight_as_lm_head_source() {
        // Qwen / Llama 3 8B: GGUF에 output.weight 있음 → 그것을 source로 선택.
        // is_tied = false. token_embd 존재 여부와 관계없이 separate 우선.
        let src = select_lm_head_source(true, true).expect("separate source must be selected");
        assert_eq!(src.source_name, LM_HEAD_SEPARATE_NAME);
        assert_eq!(src.source_name, "output.weight");
        assert!(!src.is_tied);

        // token_embd 없어도 separate만 있으면 동작 (이론적 케이스).
        let src2 = select_lm_head_source(true, false).expect("separate-only must work");
        assert_eq!(src2.source_name, LM_HEAD_SEPARATE_NAME);
        assert!(!src2.is_tied);
    }

    #[test]
    fn no_lm_head_source_returns_none() {
        // 둘 다 없는 안전망: None 반환.
        let src = select_lm_head_source(false, false);
        assert!(src.is_none());
    }

    #[test]
    fn tied_model_quantize_deterministic() {
        // tied 케이스에서 token_embd raw bytes (F16) → quantize_lm_head_to_q4_0
        // 두 번 호출 시 byte-level identical 출력.
        // separate 케이스와 동일 helper를 사용하므로 결정성이 자동으로 보장되지만,
        // tied 시나리오에서도 명시적으로 검증한다 (Sprint G-1-B fix 회귀 방지).
        use half::f16;

        let rows: usize = 4; // vocab
        let cols: usize = 64; // hidden, multiple of 32
        // Pseudo-random F16 입력 (deterministic seed; tied source는 일반적으로 F16).
        let f16_data: Vec<u16> = (0..rows * cols)
            .map(|i| f16::from_f32(((i as f32) * 0.0173).cos()).to_bits())
            .collect();
        let raw_bytes: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out1 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F16, true).unwrap();
        let out2 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F16, true).unwrap();
        assert_eq!(
            out1, out2,
            "tied lm_head Q4_0 quantize must be deterministic"
        );
        assert_eq!(out1.len(), rows * cols / 32 * 18); // 8 blocks × 18B = 144B
    }

    #[test]
    fn tied_model_lm_head_q4_0_dtype_check() {
        // tied 케이스 quantize 결과의 size invariant: rows*cols / QK4_0 * 18.
        // dtype 자체는 Vec<u8>지만, 크기로 Q4_0 layout (block 18B) 검증.
        let rows: usize = 8;
        let cols: usize = 32;
        let f32_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let raw: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_F32, true).unwrap();
        // 1 block per row × 8 rows = 8 blocks × 18B = 144B.
        assert_eq!(out.len(), 8 * 18);
        // Q4_0 block layout: 2B scale (f16) + 16B nibbles. 첫 block size 검증.
        // (raw 데이터가 0 근처라 scale도 작은 f16. byte content 자체는 결정적이지만 비교는 size로 충분.)
        assert!(
            out.len().is_multiple_of(18),
            "output must be multiple of 18B"
        );
    }

    // ── G-1-F fix: lm_head ADRENO_SOA AOS exception (INV-135 v2) ───────────────

    /// `build_variant_payload(TAG_WEIGHTS_ADRENO_SOA, ...)` 호출 시 lm_head는 SOA
    /// 변환을 skip하고 AOS bytes를 그대로 emit해야 한다 (image1d_buffer_t 한계 회피).
    /// 다른 layer weight는 SOA 변환을 그대로 적용.
    #[test]
    fn build_variant_payload_skips_soa_for_lm_head() {
        use llm_rs2::auf::q4_0_aos_to_adreno_soa;
        use llm_rs2::auf::section::TAG_WEIGHTS_ADRENO_SOA;

        // VOCAB=64, HIDDEN=128 → 64*128/32 = 256 blocks * 18B = 4608B.
        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const NUM_BLOCKS: usize = VOCAB * HIDDEN / 32;
        let lm_head_aos: Vec<u8> = (0..NUM_BLOCKS * 18).map(|i| (i % 256) as u8).collect();

        // Layer weight (FFN gate) — 32×64 = 64 blocks.
        const LAYER_ROWS: usize = 32;
        const LAYER_COLS: usize = 64;
        const LAYER_BLOCKS: usize = LAYER_ROWS * LAYER_COLS / 32;
        let layer_aos: Vec<u8> = (0..LAYER_BLOCKS * 18)
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();

        let blobs: Vec<WeightBlob> = vec![
            (
                "blk.0.ffn_gate.weight".to_owned(),
                layer_aos.clone(),
                vec![LAYER_ROWS as u64, LAYER_COLS as u64],
            ),
            (
                "output.weight".to_owned(), // = LM_HEAD_SEPARATE_NAME, lm_head 식별
                lm_head_aos.clone(),
                vec![VOCAB as u64, HIDDEN as u64],
            ),
        ];

        let payload = build_variant_payload(&blobs, TAG_WEIGHTS_ADRENO_SOA).unwrap();

        // Layer weight 부분: SOA 변환 적용.
        let (layer_q, layer_d) = q4_0_aos_to_adreno_soa(&layer_aos, LAYER_COLS, LAYER_ROWS);
        let layer_soa_len = layer_q.len() + layer_d.len();
        assert_eq!(
            layer_soa_len,
            LAYER_BLOCKS * 18,
            "SOA total size = AOS total size invariant"
        );
        assert_eq!(&payload[..layer_q.len()], &layer_q[..]);
        assert_eq!(
            &payload[layer_q.len()..layer_q.len() + layer_d.len()],
            &layer_d[..]
        );

        // lm_head 부분: AOS bytes 그대로 (SOA 변환 미적용).
        let lm_head_start = layer_soa_len;
        assert_eq!(
            &payload[lm_head_start..lm_head_start + lm_head_aos.len()],
            &lm_head_aos[..],
            "lm_head must be emitted as raw AOS bytes (no SOA transform)"
        );

        // 총 byte 길이는 N*18 invariant 유지.
        assert_eq!(payload.len(), layer_soa_len + lm_head_aos.len());
    }

    /// `build_variant_payload`는 layer weight (kind != lm_head)에 대해서는 SOA 변환을
    /// 그대로 적용. 이는 G-1-F fix가 lm_head 한정 예외임을 검증.
    #[test]
    fn build_variant_payload_applies_soa_for_layer_weights() {
        use llm_rs2::auf::q4_0_aos_to_adreno_soa;
        use llm_rs2::auf::section::TAG_WEIGHTS_ADRENO_SOA;

        const ROWS: usize = 32;
        const COLS: usize = 64;
        let aos: Vec<u8> = (0..ROWS * COLS / 32 * 18)
            .map(|i| (i % 256) as u8)
            .collect();

        let blobs: Vec<WeightBlob> = vec![(
            "blk.5.attn_v.weight".to_owned(),
            aos.clone(),
            vec![ROWS as u64, COLS as u64],
        )];

        let payload = build_variant_payload(&blobs, TAG_WEIGHTS_ADRENO_SOA).unwrap();
        let (q, d) = q4_0_aos_to_adreno_soa(&aos, COLS, ROWS);

        assert_eq!(&payload[..q.len()], &q[..]);
        assert_eq!(&payload[q.len()..q.len() + d.len()], &d[..]);
        assert_eq!(payload.len(), q.len() + d.len());
        // SOA-converted bytes는 원본 AOS bytes와 다름이 정상 (sanity).
        assert_ne!(&payload[..aos.len()], &aos[..]);
    }

    #[test]
    fn round_trip_strip_via_auf_api() {
        use llm_rs2::auf::{AufMeta, AufTokenizer, AufWriter, strip_bytes};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };
        let original = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 128])
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![2u8; 64])
            .build()
            .unwrap();

        // ADRENO만 유지
        let stripped = strip_bytes(&original, &[TAG_WEIGHTS_ADRENO_SOA]).unwrap();

        // stripped에서 CPU_AOS section이 없어야 함
        use llm_rs2::auf::header::AufHeader;
        use llm_rs2::auf::section::SectionTable;
        let hdr = AufHeader::from_bytes(&stripped).unwrap();
        let tbl = SectionTable::from_bytes(
            &stripped[hdr.section_table_offset as usize..],
            hdr.section_count,
        )
        .unwrap();
        let tags: Vec<_> = tbl.entries.iter().map(|e| e.tag().to_owned()).collect();
        assert!(tags.contains(&"WEIGHTS_ADRENO_SOA".to_owned()));
        assert!(!tags.contains(&"WEIGHTS_CPU_AOS".to_owned()));
        assert_eq!(hdr.section_count, 4); // META + TOKENIZER + TENSOR_INDEX + ADRENO_SOA
    }
}
