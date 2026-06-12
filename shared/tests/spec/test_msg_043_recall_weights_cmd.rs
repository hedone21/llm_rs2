//! MSG-043: EngineCommand::RecallWeights serde + dispatch contract.
//!
//! 커버: MSG-043 RecallWeights round-trip + 기존 변형 serde 무회귀.
//! - `{"type":"recall_weights","ratio":1.0}` 직렬화 계약.
//! - `target_dtype` 필드 없음(방향이 dtype을 고정, ENG-ALG-240).
//! - 기존 15종 변형 serde 무회귀.

use llm_shared::EngineCommand;

// ── MSG-043: RecallWeights serde round-trip ──────────────────────────────────

#[test]
fn test_msg_043_recall_weights_serde_roundtrip() {
    let cmd = EngineCommand::RecallWeights { ratio: 1.0 };
    let json = serde_json::to_string(&cmd).unwrap();
    // wire 키 검증
    assert!(
        json.contains("\"type\":\"recall_weights\""),
        "expected recall_weights type key, got: {json}"
    );
    assert!(
        json.contains("\"ratio\":1.0"),
        "expected ratio=1.0, got: {json}"
    );
    // target_dtype 필드 없음 (ENG-ALG-240: 방향이 dtype을 고정)
    assert!(
        !json.contains("target_dtype"),
        "RecallWeights must NOT have target_dtype field, got: {json}"
    );

    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    match back {
        EngineCommand::RecallWeights { ratio } => {
            assert!((ratio - 1.0).abs() < f32::EPSILON, "ratio round-trip");
        }
        _ => panic!("Expected RecallWeights, got {back:?}"),
    }
}

#[test]
fn test_msg_043_recall_weights_partial_ratio() {
    let cmd = EngineCommand::RecallWeights { ratio: 0.5 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"recall_weights\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    match back {
        EngineCommand::RecallWeights { ratio } => {
            assert!((ratio - 0.5).abs() < f32::EPSILON);
        }
        _ => panic!("Expected RecallWeights"),
    }
}

// ── 기존 변형 serde 무회귀 ────────────────────────────────────────────────────

#[test]
fn test_msg_043_existing_variants_unaffected() {
    // SwapWeights 무회귀
    let cmd = llm_shared::EngineCommand::SwapWeights {
        ratio: 0.5,
        target_dtype: llm_shared::DtypeTag::Q4_0,
    };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"swap_weights\""));
    assert!(json.contains("\"target_dtype\":\"q4_0\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(
        back,
        EngineCommand::SwapWeights {
            ratio: _,
            target_dtype: llm_shared::DtypeTag::Q4_0
        }
    ));

    // RestoreDefaults 무회귀 (INV-192: RestoreDefaults는 swap/recall 무발화)
    let cmd = EngineCommand::RestoreDefaults;
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"restore_defaults\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::RestoreDefaults));

    // KvEvictH2o 무회귀
    let cmd = EngineCommand::KvEvictH2o { keep_ratio: 0.5 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"kv.evict_h2o\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::KvEvictH2o { .. }));

    // Throttle 무회귀
    let cmd = EngineCommand::Throttle { delay_ms: 30 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"throttle\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::Throttle { delay_ms: 30 }));
}

#[test]
fn test_msg_043_recall_weights_deserialize_from_literal() {
    // 최소 JSON — wire format 계약 고정
    let json = r#"{"type":"recall_weights","ratio":0.75}"#;
    let back: EngineCommand = serde_json::from_str(json).unwrap();
    match back {
        EngineCommand::RecallWeights { ratio } => {
            assert!((ratio - 0.75).abs() < f32::EPSILON);
        }
        _ => panic!("Expected RecallWeights from literal"),
    }
}
