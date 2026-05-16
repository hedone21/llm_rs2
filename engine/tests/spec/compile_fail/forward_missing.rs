//! INV-LAYER-007 negative: build() must not compile without a Forward.

use llm_rs2::session::DecodeLoopBuilder;

fn main() {
    let _loop = DecodeLoopBuilder::new().build();
}
