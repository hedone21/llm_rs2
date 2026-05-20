//! Phase 4-A G_unit: `session::batch::helpers`의 JSONL 파싱 / prompt
//! resolution 동치 검증. batch runner 본문은 integration 성격이라
//! 디바이스 G2 게이트(2-entry baseline diff)로 검증한다.

use llm_rs2::session::batch::helpers::{load_prompt_batch, resolve_prompt};

#[test]
fn load_prompt_batch_parses_valid_jsonl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("batch.jsonl");
    std::fs::write(
        &path,
        "# comment ignored\n\
         {\"id\": \"q1\", \"prompt\": \"hello\"}\n\
         \n\
         {\"id\": \"q2\", \"prompt_file\": \"some/path.txt\"}\n",
    )
    .unwrap();

    let entries = load_prompt_batch(path.to_str().unwrap()).unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].id, "q1");
    assert_eq!(entries[0].prompt.as_deref(), Some("hello"));
    assert!(entries[0].prompt_file.is_none());
    assert_eq!(entries[1].id, "q2");
    assert!(entries[1].prompt.is_none());
    assert_eq!(entries[1].prompt_file.as_deref(), Some("some/path.txt"));
}

#[test]
fn load_prompt_batch_rejects_invalid_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.jsonl");
    std::fs::write(&path, "{not valid json}\n").unwrap();
    assert!(load_prompt_batch(path.to_str().unwrap()).is_err());
}

#[test]
fn load_prompt_batch_empty_file_returns_empty_vec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.jsonl");
    std::fs::write(&path, "# only comments\n\n").unwrap();
    let entries = load_prompt_batch(path.to_str().unwrap()).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn resolve_prompt_returns_inline_text() {
    let entries = load_prompt_batch_inline(r#"{"id":"a","prompt":"inline"}"#);
    let text = resolve_prompt(&entries[0]).unwrap();
    assert_eq!(text, "inline");
}

#[test]
fn resolve_prompt_reads_prompt_file() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_path = dir.path().join("p.txt");
    std::fs::write(&prompt_path, "from file").unwrap();
    let jsonl_path = dir.path().join("b.jsonl");
    std::fs::write(
        &jsonl_path,
        format!(
            "{{\"id\":\"a\",\"prompt_file\":\"{}\"}}",
            prompt_path.display()
        ),
    )
    .unwrap();
    let entries = load_prompt_batch(jsonl_path.to_str().unwrap()).unwrap();
    let text = resolve_prompt(&entries[0]).unwrap();
    assert_eq!(text, "from file");
}

#[test]
fn resolve_prompt_errors_when_both_missing() {
    let entries = load_prompt_batch_inline(r#"{"id":"a"}"#);
    let err = resolve_prompt(&entries[0]).unwrap_err();
    assert!(format!("{err}").contains("needs 'prompt' or 'prompt_file'"));
}

fn load_prompt_batch_inline(
    jsonl: &str,
) -> Vec<llm_rs2::session::batch::helpers::PromptBatchEntry> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tmp.jsonl");
    std::fs::write(&path, jsonl).unwrap();
    load_prompt_batch(path.to_str().unwrap()).unwrap()
}
