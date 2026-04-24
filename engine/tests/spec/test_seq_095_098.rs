//! SEQ-095 ~ SEQ-098: QCF Request/Estimate sequence (Engine side)
//!
//! SEQ-095: Manager sends Directive([RequestQcf])
//! SEQ-096: Engine responds Ok, then sends separate QcfEstimate
//! SEQ-097: (Manager side — tested in manager/tests/spec/)
//! SEQ-098: (Manager side — timeout fallback)

use std::collections::HashMap;
use std::time::Duration;

use llm_shared::{CommandResult, EngineCommand, EngineMessage, QcfEstimate};

use super::helpers::{empty_snap, make_executor, send_directive};

// ═══════════════════════════════════════════════════════════════
// SEQ-095: RequestQcf Directive 수신 및 처리
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_095_request_qcf_returns_ok() {
    // SEQ-095: Engine receives Directive([RequestQcf]) and returns Ok
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::RequestQcf]);
    let plan = executor.poll(&empty_snap());

    // Verify plan flag
    assert!(plan.request_qcf, "RequestQcf should set plan flag");

    // Verify Response
    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(resp.seq_id, 1);
            assert_eq!(resp.results.len(), 1);
            assert!(matches!(resp.results[0], CommandResult::Ok));
        }
        _ => panic!("Expected Response, got {:?}", msg),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: Response(Ok) 후 별도 QcfEstimate 전송
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_send_qcf_estimate_after_response() {
    // SEQ-096: Engine sends Response first, then QcfEstimate as separate message
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 5, vec![EngineCommand::RequestQcf]);
    let _plan = executor.poll(&empty_snap());

    // Message 1: Response(Ok)
    let msg1 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg1, EngineMessage::Response(_)),
        "First message should be Response"
    );

    // Simulate QCF computation and send (as generate.rs would do)
    let mut estimates = HashMap::new();
    estimates.insert("kv_evict_sliding".to_string(), 0.15);
    estimates.insert("kv_evict_h2o".to_string(), 0.22);
    executor.send_qcf_estimate(QcfEstimate {
        estimates: estimates.clone(),
        layer_swap: None,
    });

    // Message 2: QcfEstimate
    let msg2 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg2 {
        EngineMessage::QcfEstimate(qcf) => {
            assert_eq!(qcf.estimates.len(), 2);
            assert!((qcf.estimates["kv_evict_sliding"] - 0.15).abs() < f32::EPSILON);
            assert!((qcf.estimates["kv_evict_h2o"] - 0.22).abs() < f32::EPSILON);
        }
        _ => panic!("Expected QcfEstimate, got {:?}", msg2),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: QcfEstimate 빈 map (캐시 비어있을 때)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_empty_qcf_estimate_when_cache_empty() {
    // MSG-086: Engine sends empty estimates when no action computable
    let (executor, _tx, rx) = make_executor();

    executor.send_qcf_estimate(QcfEstimate {
        estimates: HashMap::new(),
        layer_swap: None,
    });

    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::QcfEstimate(qcf) => {
            assert!(qcf.estimates.is_empty());
        }
        _ => panic!("Expected QcfEstimate"),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-095: RequestQcf와 다른 커맨드 혼합
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_095_request_qcf_mixed_with_other_commands() {
    // RequestQcf can be included with other commands in same Directive
    let (mut executor, tx, rx) = make_executor();
    send_directive(
        &tx,
        2,
        vec![
            EngineCommand::Throttle { delay_ms: 50 },
            EngineCommand::RequestQcf,
        ],
    );
    let plan = executor.poll(&empty_snap());

    assert!(plan.request_qcf);
    assert_eq!(plan.throttle_delay_ms, 50);

    let msg = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    match msg {
        EngineMessage::Response(resp) => {
            assert_eq!(resp.seq_id, 2);
            assert_eq!(resp.results.len(), 2);
            assert!(matches!(resp.results[0], CommandResult::Ok));
            assert!(matches!(resp.results[1], CommandResult::Ok));
        }
        _ => panic!("Expected Response"),
    }
}

// ═══════════════════════════════════════════════════════════════
// SEQ-096: Response와 QcfEstimate 순서 보장
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_096_message_ordering_response_before_estimate() {
    // SEQ-096: Response MUST be sent before QcfEstimate
    let (mut executor, tx, rx) = make_executor();
    send_directive(&tx, 7, vec![EngineCommand::RequestQcf]);
    let _plan = executor.poll(&empty_snap());

    // Send QcfEstimate (simulating generate.rs behavior after poll)
    let mut estimates = HashMap::new();
    estimates.insert("kv_evict_sliding".to_string(), 0.1);
    executor.send_qcf_estimate(QcfEstimate {
        estimates,
        layer_swap: None,
    });

    // Verify order: Response first, then QcfEstimate
    let msg1 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg1, EngineMessage::Response(ref r) if r.seq_id == 7),
        "First message must be Response"
    );

    let msg2 = rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(
        matches!(msg2, EngineMessage::QcfEstimate(_)),
        "Second message must be QcfEstimate"
    );
}

// ═══════════════════════════════════════════════════════════════
// SEQ-095: request_qcf 기본값은 false
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_seq_095_request_qcf_default_false() {
    // Non-RequestQcf commands should not set request_qcf flag
    let (mut executor, tx, _rx) = make_executor();
    send_directive(&tx, 1, vec![EngineCommand::Throttle { delay_ms: 10 }]);
    let plan = executor.poll(&empty_snap());
    assert!(!plan.request_qcf);
}
