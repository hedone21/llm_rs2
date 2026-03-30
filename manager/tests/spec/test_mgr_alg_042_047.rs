//! MGR-ALG-042 ~ MGR-ALG-047: ReliefEstimator Spec 테스트
//!
//! OnlineLinearEstimator의 초기 default relief, 학습 수렴,
//! 액션 독립성, observation count, save/load roundtrip을 검증한다.

use llm_manager::relief::linear::OnlineLinearEstimator;
use llm_manager::relief::ReliefEstimator;
use llm_manager::types::{ActionId, FEATURE_DIM, FeatureVector, ReliefVector};

fn make_state(values: [f32; FEATURE_DIM]) -> FeatureVector {
    FeatureVector { values }
}

fn zero_state() -> FeatureVector {
    FeatureVector::zeros()
}

fn simple_state() -> FeatureVector {
    let mut values = [0.0f32; FEATURE_DIM];
    values[0] = 0.5; // kv_occupancy
    values[1] = 1.0; // is_gpu
    values[2] = 0.3; // token_progress
    FeatureVector { values }
}

fn relief(compute: f32, memory: f32, thermal: f32, latency: f32) -> ReliefVector {
    ReliefVector {
        compute,
        memory,
        thermal,
        latency,
    }
}

// ── MGR-ALG-042 / MGR-084: 초기 predict는 default relief 반환 ──────

/// MGR-ALG-042: 관측 없는 초기 predict는 도메인 기반 default relief를 반환한다.
#[test]
fn test_mgr_alg_042_predict_initial_returns_default_relief() {
    let estimator = OnlineLinearEstimator::default_config();
    let state = simple_state();
    let pred = estimator.predict(&ActionId::KvEvictSliding, &state);
    // KvEvictSliding default: compute=0.0, memory=0.7, thermal=0.0, latency=0.0
    assert!(
        pred.compute.abs() < 1e-6,
        "initial compute should be 0.0, got {}",
        pred.compute
    );
    assert!(
        (pred.memory - 0.7).abs() < 1e-6,
        "initial memory should be 0.7, got {}",
        pred.memory
    );
    assert!(
        pred.thermal.abs() < 1e-6,
        "initial thermal should be 0.0, got {}",
        pred.thermal
    );
    assert!(
        pred.latency.abs() < 1e-6,
        "initial latency should be 0.0, got {}",
        pred.latency
    );
}

// ── MGR-ALG-043 / MGR-086: 각 액션의 default relief 값 ─────────────

/// MGR-ALG-043: 각 액션별 default relief가 올바르게 매핑된다.
#[test]
fn test_mgr_alg_043_default_relief_values() {
    let estimator = OnlineLinearEstimator::default_config();
    let state = simple_state();

    let sw = estimator.predict(&ActionId::SwitchHw, &state);
    assert!(
        (sw.compute - 0.5).abs() < 1e-6,
        "SwitchHw compute={}",
        sw.compute
    );
    assert!(
        (sw.thermal - 0.3).abs() < 1e-6,
        "SwitchHw thermal={}",
        sw.thermal
    );
    assert!(
        (sw.latency - (-0.1)).abs() < 1e-6,
        "SwitchHw latency={}",
        sw.latency
    );

    let thr = estimator.predict(&ActionId::Throttle, &state);
    assert!(
        (thr.compute - 0.3).abs() < 1e-6,
        "Throttle compute={}",
        thr.compute
    );
    assert!(
        (thr.latency - (-0.3)).abs() < 1e-6,
        "Throttle latency={}",
        thr.latency
    );

    let ls = estimator.predict(&ActionId::LayerSkip, &state);
    assert!(
        (ls.compute - 0.3).abs() < 1e-6,
        "LayerSkip compute={}",
        ls.compute
    );
    assert!(
        (ls.latency - (-0.2)).abs() < 1e-6,
        "LayerSkip latency={}",
        ls.latency
    );
}

// ── MGR-083: 액션 간 독립성 ────────────────────────────────────────

/// MGR-083: action A의 observe가 action B의 predict에 영향을 미치지 않는다.
#[test]
fn test_mgr_083_multiple_actions_independent() {
    let mut estimator = OnlineLinearEstimator::default_config();
    let state = simple_state();
    let actual = relief(0.5, 0.7, 0.2, -0.1);

    for _ in 0..20 {
        estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
    }

    // SwitchHw는 학습하지 않았으므로 default relief 반환
    let pred_b = estimator.predict(&ActionId::SwitchHw, &state);
    assert!(
        (pred_b.compute - 0.5).abs() < 1e-6,
        "untrained SwitchHw should return default compute=0.5, got {}",
        pred_b.compute
    );
    assert!(
        pred_b.memory.abs() < 1e-6,
        "untrained SwitchHw should return default memory=0.0, got {}",
        pred_b.memory
    );
}

// ── MGR-ALG-044: 학습 후 수렴 ──────────────────────────────────────

/// MGR-ALG-044: observe 후 predict 결과가 actual 방향으로 수렴한다.
#[test]
fn test_mgr_alg_044_observe_and_predict() {
    let mut estimator = OnlineLinearEstimator::default_config();
    let state = simple_state();
    let actual = relief(0.6, 0.8, 0.3, -0.2);

    for _ in 0..10 {
        estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
    }

    let pred = estimator.predict(&ActionId::KvEvictSliding, &state);
    assert!(
        pred.compute > 0.0,
        "compute should be positive after training, got {}",
        pred.compute
    );
    assert!(
        pred.memory > 0.0,
        "memory should be positive after training, got {}",
        pred.memory
    );
    assert!(
        pred.thermal > 0.0,
        "thermal should be positive after training, got {}",
        pred.thermal
    );
    assert!(
        pred.latency < 0.0,
        "latency should be negative after training, got {}",
        pred.latency
    );
}

/// MGR-ALG-044: 동일 state에 반복 observe 후 예측이 actual에 점차 수렴.
#[test]
fn test_mgr_alg_044_convergence_direction() {
    let mut estimator = OnlineLinearEstimator::default_config();
    let state = make_state([1.0f32; FEATURE_DIM]);
    let actual = relief(0.8, 0.5, 0.0, -0.4);

    for _ in 0..30 {
        estimator.observe(&ActionId::KvEvictH2o, &state, &actual);
    }

    let pred = estimator.predict(&ActionId::KvEvictH2o, &state);
    assert!(
        (pred.compute - actual.compute).abs() < actual.compute * 0.5,
        "compute did not converge: pred={}, actual={}",
        pred.compute,
        actual.compute
    );
    assert!(
        (pred.memory - actual.memory).abs() < actual.memory * 0.5,
        "memory did not converge: pred={}, actual={}",
        pred.memory,
        actual.memory
    );
}

// ── MGR-085: observation_count ──────────────────────────────────────

/// MGR-085: observe 호출 횟수와 observation_count가 일치한다.
#[test]
fn test_mgr_085_observation_count() {
    let mut estimator = OnlineLinearEstimator::default_config();
    let state = zero_state();
    let actual = relief(0.1, 0.2, 0.0, 0.0);

    assert_eq!(estimator.observation_count(&ActionId::Throttle), 0);

    for i in 1..=7 {
        estimator.observe(&ActionId::Throttle, &state, &actual);
        assert_eq!(
            estimator.observation_count(&ActionId::Throttle),
            i,
            "count should be {i}"
        );
    }

    // 다른 액션은 별도 카운트
    assert_eq!(estimator.observation_count(&ActionId::SwitchHw), 0);
}

// ── MGR-087 / MGR-ALG-047: save/load roundtrip ─────────────────────

/// MGR-087: save -> load -> predict 결과가 동일하다 (roundtrip).
#[test]
fn test_mgr_alg_047_save_load_roundtrip() {
    let mut estimator = OnlineLinearEstimator::default_config();
    let state = simple_state();
    let actual = relief(0.4, 0.6, 0.1, -0.3);

    for _ in 0..5 {
        estimator.observe(&ActionId::KvEvictSliding, &state, &actual);
        estimator.observe(&ActionId::SwitchHw, &state, &actual);
    }

    let pred_before = estimator.predict(&ActionId::KvEvictSliding, &state);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    estimator.save(tmp.path()).expect("save should succeed");

    let mut estimator2 = OnlineLinearEstimator::default_config();
    estimator2.load(tmp.path()).expect("load should succeed");

    let pred_after = estimator2.predict(&ActionId::KvEvictSliding, &state);

    assert!(
        (pred_before.compute - pred_after.compute).abs() < 1e-5,
        "compute mismatch: {} vs {}",
        pred_before.compute,
        pred_after.compute
    );
    assert!(
        (pred_before.memory - pred_after.memory).abs() < 1e-5,
        "memory mismatch: {} vs {}",
        pred_before.memory,
        pred_after.memory
    );
    assert!(
        (pred_before.latency - pred_after.latency).abs() < 1e-5,
        "latency mismatch: {} vs {}",
        pred_before.latency,
        pred_after.latency
    );
}
