//! Spec test for INV-155 (`pkg_free_op_impl` 정상화 / leak 부재).
//!
//! INV-155 본 검증은 디바이스 microbench (`microbench_qnn_oppkg_leak`,
//! 100회 register/free 후 last-50 VmRSS slope < 1 KB/iter)에서 수행한다.
//! host에서는 reverse-mapping table API의 핵심 불변식만 검증한다:
//!   1. `pkg_free_op_impl(NULL)` → SUCCESS (idempotent / null-safe).
//!   2. `build_op_state` 호출 후 `STATE_MAP` 엔트리 +1, free 호출 후 0.
//!
//! 이 두 검증으로 device microbench가 leak slope를 의미 있게 측정 가능함을
//! 보장한다 (drop이 실제로 일어나야 slope ≈ 0이 나오므로).

use qnn_oppkg::__test_support::{build_add_layout_n, call_pkg_free_op_impl, state_map_len};
use qnn_oppkg::ensure_static;

const QNN_SUCCESS: u64 = 0;

#[test]
fn pkg_free_op_impl_with_null_returns_success() {
    // INV-155 idempotency: null 입력 시 SUCCESS 반환 + crash 없음.
    let rc = call_pkg_free_op_impl(std::ptr::null_mut());
    assert_eq!(rc, QNN_SUCCESS, "pkg_free_op_impl(NULL) should be SUCCESS");
}

#[test]
fn build_op_state_register_and_free_drops_entry() {
    // INV-155 round-trip: build → STATE_MAP +1 → free → STATE_MAP 회복.
    // 동시 다른 테스트들도 STATE_MAP을 건드리므로 absolute count가 아닌
    // delta로 검증한다.
    ensure_static();

    use qnn_oppkg::__test_support::raw_build_op_state_for_add;

    let before = state_map_len();
    let layout = build_add_layout_n(64).expect("layout build");
    let op_ptr = raw_build_op_state_for_add(layout).expect("build_op_state");
    assert!(!op_ptr.is_null(), "build_op_state should return non-null");

    let after_build = state_map_len();
    assert_eq!(
        after_build,
        before + 1,
        "build_op_state should register exactly one entry"
    );

    // free
    let rc = call_pkg_free_op_impl(op_ptr);
    assert_eq!(rc, QNN_SUCCESS, "pkg_free_op_impl should return SUCCESS");

    let after_free = state_map_len();
    assert_eq!(
        after_free, before,
        "pkg_free_op_impl should drop the entry it owned"
    );

    // 두 번째 free는 no-op (이미 제거됨). idempotent.
    let rc2 = call_pkg_free_op_impl(op_ptr);
    assert_eq!(
        rc2, QNN_SUCCESS,
        "pkg_free_op_impl should be idempotent on stale pointer"
    );
    assert_eq!(state_map_len(), before, "second free must not change len");
}
