//! Chat REPL의 inner decode loop 종료 조건 trait + helper.
//! Phase 4-5-c. chat REPL은 4-5-e에서 ChatStopCondition을 owned 보유하여
//! DecodeLoop::run_until_stop에 전달.

use crate::session::chat_template::ChatTemplate;
use tokenizers::Tokenizer;

/// Decode loop 종료 조건. `should_stop`이 true를 반환하면 loop를 즉시 종료한다.
///
/// Stop token 체크 위치: sampled 직후 호출. true이면 즉시 break하므로
/// stop token은 KV에 baking되지 않는다. chat REPL에서 EOT는 별도 decode_step으로
/// baking(4-5-e에서 처리).
pub trait StopCondition {
    /// `sampled` token이 방금 emit되었고 KV pos가 `pos`인 시점에 호출.
    /// true 반환 시 decode loop 종료. sampled token은 KV에 baking되지 않았음.
    fn should_stop(&self, sampled: u32, pos: usize) -> bool;
}

/// Chat 모드 전용 stop condition.
/// - stop_ids 중 하나가 sampled되면 true
/// - pos가 max_pos 이상이면 true (overflow 안전망)
pub struct ChatStopCondition {
    stop_ids: Vec<u32>,
    max_pos: usize,
}

impl ChatStopCondition {
    /// `stop_ids`는 내부에서 sort + dedup하므로 순서 무관.
    pub fn new(stop_ids: Vec<u32>, max_pos: usize) -> Self {
        let mut ids = stop_ids;
        ids.sort_unstable();
        ids.dedup();
        Self {
            stop_ids: ids,
            max_pos,
        }
    }
}

impl StopCondition for ChatStopCondition {
    fn should_stop(&self, sampled: u32, pos: usize) -> bool {
        if pos >= self.max_pos {
            return true;
        }
        self.stop_ids.binary_search(&sampled).is_ok()
    }
}

/// 토크나이저로 literal token들을 토큰 ID로 변환.
///
/// `required=true`인데 literal에 대한 ID를 찾지 못하면 Err.
/// `required=false`인 경우 없는 literal은 조용히 스킵.
///
/// generate.rs의 `resolve_token_ids` (l.9805~9824)와 동일 시그니처 — 4-5-f에서
/// generate.rs의 사본은 삭제하고 이 버전을 사용하게 된다.
pub fn resolve_token_ids(
    tokenizer: &Tokenizer,
    literals: &[&'static str],
    required: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(literals.len());
    for lit in literals {
        match tokenizer.token_to_id(lit) {
            Some(id) => out.push(id),
            None if required => {
                anyhow::bail!(
                    "tokenizer is missing required special token `{}`. \
                     Make sure tokenizer.json has it registered as an added_token.",
                    lit
                );
            }
            None => {}
        }
    }
    Ok(out)
}

/// Chat stop_ids 빌드 헬퍼.
///
/// `template.stop_token_literals()` + `eos_token_id`를 합산하여
/// sorted dedup Vec<u32>를 반환한다.
///
/// generate.rs:9870~9881의 stop_ids 빌드 로직을 이관.
pub fn build_chat_stop_ids(
    template: &ChatTemplate,
    tokenizer: &Tokenizer,
    eos_token_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let lits = template.stop_token_literals();
    if lits.is_empty() {
        anyhow::bail!("chat template has no stop token literals");
    }
    let mut ids = resolve_token_ids(tokenizer, &lits[..1], true)?;
    ids.extend(resolve_token_ids(tokenizer, &lits[1..], false)?);
    ids.push(eos_token_id);
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_stop_returns_true_on_stop_id_match() {
        let cond = ChatStopCondition::new(vec![128009, 128001], 2048);
        assert!(cond.should_stop(128009, 10));
        assert!(cond.should_stop(128001, 10));
        assert!(!cond.should_stop(42, 10));
    }

    #[test]
    fn should_stop_returns_true_on_max_pos_overflow() {
        let cond = ChatStopCondition::new(vec![128009], 100);
        // pos < max_pos, non-stop token → false
        assert!(!cond.should_stop(1, 99));
        // pos == max_pos → true (안전망)
        assert!(cond.should_stop(1, 100));
        // pos > max_pos → true
        assert!(cond.should_stop(1, 200));
    }

    #[test]
    fn resolve_token_ids_errors_when_required_but_empty() {
        // tokenizers::Tokenizer::from_pretrained은 네트워크 없이 사용 불가.
        // 더미 vocab JSON으로 Tokenizer를 직접 생성하여 검증.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "vocab": {"hello": 0, "world": 1},
                "merges": [],
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false
            }
        }"#;
        use std::str::FromStr;
        let tokenizer = Tokenizer::from_str(json).expect("tokenizer parse");

        // required=true이고 토큰이 없으면 Err
        let result = resolve_token_ids(&tokenizer, &["<|eot_id|>"], true);
        assert!(result.is_err(), "expected Err for missing required token");

        // required=false이면 스킵하고 Ok
        let result = resolve_token_ids(&tokenizer, &["<|eot_id|>"], false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
