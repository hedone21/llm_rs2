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
    compute_source_hash, open,
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

    /// 진행 로그 출력 억제
    #[arg(long)]
    quiet: bool,
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
    })
}

/// Q4_0 블록 raw bytes를 SOA 형식(q_buf + d_buf)으로 변환한다.
///
/// GGUF Q4_0 블록 (18B per block, QK=32):
///   [0..2)  d: f16 (scale)
///   [2..18) qs: [u8; 16] (nibbles, 32 values packed)
///
/// SOA 출력:
///   q_buf: 블록별 nibbles 연속 배치 (n_blocks * 16B)
///   d_buf: 블록별 scale f16 연속 배치 (n_blocks * 2B)
fn q4_0_aos_to_soa(blocks: &[u8]) -> (Vec<u8>, Vec<u8>) {
    const BLOCK_SIZE: usize = 18; // 2 (f16 scale) + 16 (nibbles)
    let n_blocks = blocks.len() / BLOCK_SIZE;
    let mut q_buf = Vec::with_capacity(n_blocks * 16);
    let mut d_buf = Vec::with_capacity(n_blocks * 2);
    for i in 0..n_blocks {
        let off = i * BLOCK_SIZE;
        d_buf.extend_from_slice(&blocks[off..off + 2]); // f16 scale
        q_buf.extend_from_slice(&blocks[off + 2..off + 18]); // nibbles
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
    let quiet = args.quiet;

    if !quiet {
        eprintln!(
            "[auf-tool] Build: {} → {}",
            args.input.display(),
            args.output.display()
        );
        eprintln!("[auf-tool] Variants: {:?}", variant_tags);
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
    if !quiet {
        eprint!("[auf-tool] Extracting weight tensors...");
    }
    let tensor_blobs = extract_weight_blobs(&gguf, &config, quiet)?;
    if !quiet {
        eprintln!(" {} tensors extracted", tensor_blobs.len());
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

/// `WeightBlob` 목록 추출.
///
/// GGUF는 innermost-first로 dims를 저장하므로, 여기서 reverse하여 logical order로 반환한다.
/// AUF TensorEntry.shape에는 logical order(outermost-first)로 저장하며,
/// reader(`secondary_mmap.rs`)에서 다시 reverse하여 GGUF order로 복원한다.
fn extract_weight_blobs(
    gguf: &llm_rs2::models::loader::gguf::GgufFile,
    config: &llm_rs2::models::config::ModelConfig,
    quiet: bool,
) -> Result<Vec<WeightBlob>> {
    let mut blobs: Vec<WeightBlob> = Vec::new();

    // cross-layer tensors
    let cross_names = ["token_embd.weight", "output_norm.weight", "output.weight"];
    for &name in &cross_names {
        if let Some(info) = gguf.find_tensor(name) {
            let data = gguf.tensor_data(info);
            // GGUF dims는 innermost-first → reversed = outermost-first (logical)
            let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
            blobs.push((name.to_owned(), data.to_vec(), shape_logical));
        }
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

    Ok(blobs)
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
            // SOA: 모든 Q4_0 weight를 q_buf + d_buf 분리하여 연속 배치
            // 비-Q4_0 tensor (F16 norm 등)는 그대로 포함
            let mut out = Vec::new();
            for (_name, bytes, _shape) in blobs {
                // Q4_0 식별: 블록 크기 18B (정확한 배수인지 확인)
                if bytes.len() % 18 == 0 && bytes.len() >= 18 {
                    // Q4_0 SOA 변환
                    let (q_buf, d_buf) = q4_0_aos_to_soa(bytes);
                    // q_buf 먼저, d_buf 다음 (section 내 연속)
                    out.extend_from_slice(&q_buf);
                    out.extend_from_slice(&d_buf);
                } else {
                    // F16/F32 tensor 그대로
                    out.extend_from_slice(bytes);
                }
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
    println!("  capability_opt   : {:#018x}", h.capability_optional);
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
