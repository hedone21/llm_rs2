//! Spec tests for INV-152 (OPS.len() == numOperations) and INV-153 (op_types unique).

use qnn_oppkg::{OPS, ensure_static};

#[test]
fn ops_len_equals_num_operations() {
    let info = ensure_static();
    let num_ops = info.info.numOperations;
    assert_eq!(
        OPS.len() as u32,
        num_ops,
        "INV-152: OPS.len() ({}) must equal Info_t.numOperations ({})",
        OPS.len(),
        num_ops
    );
}

#[test]
fn op_types_are_unique() {
    let mut set = std::collections::HashSet::new();
    for d in OPS {
        assert!(
            set.insert(d.op_type),
            "INV-153: duplicate op_type registered: {}",
            d.op_type
        );
    }
}
