//! Chat mode input multiplexer (stdin + optional Unix domain socket).
//!
//! Provides a single `Receiver<ChatInput>` that merges newline-delimited
//! messages from stdin and (when enabled) from a Unix socket listener.
//! Reply bytes from the model can be streamed back to the same socket
//! connection the message arrived on.

#[cfg(unix)]
pub type ChatReplyWriter = std::sync::Arc<std::sync::Mutex<std::os::unix::net::UnixStream>>;

#[cfg(not(unix))]
pub type ChatReplyWriter = std::sync::Arc<std::sync::Mutex<()>>;

pub enum ChatInput {
    Line(String, Option<ChatReplyWriter>),
    Eof,
}

/// Write bytes to the reply stream if present. Errors are swallowed — a
/// client that has closed its end should not abort the model loop.
pub fn write_reply_bytes(reply: Option<&ChatReplyWriter>, bytes: &[u8]) {
    #[cfg(unix)]
    if let Some(arc) = reply {
        use std::io::Write;
        if let Ok(mut s) = arc.lock() {
            let _ = s.write_all(bytes);
            let _ = s.flush();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (reply, bytes);
    }
}

/// Delimit end-of-turn on the reply stream with a 0x04 (EOT) byte. Does
/// not shut down the write side — the stream stays open so the next user
/// message on the same connection can receive its own reply.
pub fn finish_reply_stream(reply: Option<&ChatReplyWriter>) {
    #[cfg(unix)]
    if let Some(arc) = reply {
        use std::io::Write;
        if let Ok(mut s) = arc.lock() {
            let _ = s.write_all(&[0x04]);
            let _ = s.flush();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = reply;
    }
}

/// Spawn a stdin reader thread that forwards each newline-delimited line
/// into the given sender. On EOF or error, sends one `ChatInput::Eof` and
/// exits.
pub fn spawn_stdin_reader(tx: std::sync::mpsc::Sender<ChatInput>) {
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        loop {
            let mut buf = String::new();
            match reader.read_line(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(ChatInput::Eof);
                    break;
                }
                Ok(_) => {
                    if tx.send(ChatInput::Line(buf, None)).is_err() {
                        break;
                    }
                }
                Err(_) => {
                    let _ = tx.send(ChatInput::Eof);
                    break;
                }
            }
        }
    });
}

/// Bind a Unix socket listener and spawn an accept loop. Each accepted
/// connection gets its own reader thread that forwards newline-delimited
/// lines into `tx`, tagged with a writer handle pointing at the same
/// connection so replies can stream back. Does not consume stdin.
#[cfg(unix)]
pub fn spawn_socket_listener(
    tx: std::sync::mpsc::Sender<ChatInput>,
    path: &str,
) -> anyhow::Result<()> {
    use std::os::unix::net::UnixListener;
    let path = path.to_string();
    // Remove any stale socket file before binding.
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)
        .map_err(|e| anyhow::anyhow!("failed to bind Unix socket at {}: {}", path, e))?;
    eprintln!("[Chat] Listening for socket input at {}", path);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(_) => continue,
            };
            let writer_clone = match stream.try_clone() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let stream_arc = std::sync::Arc::new(std::sync::Mutex::new(writer_clone));
            let conn_tx = tx.clone();
            let reader_stream = stream;
            std::thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(reader_stream);
                for line in reader.lines().map_while(Result::ok) {
                    if conn_tx
                        .send(ChatInput::Line(line, Some(stream_arc.clone())))
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
    });
    Ok(())
}

/// Spawn the stdin reader and (optionally) a Unix socket listener. Both
/// push into the returned channel. This is the main production entry —
/// use `spawn_stdin_reader` / `spawn_socket_listener` directly for tests
/// or custom compositions.
pub fn spawn_chat_input_sources(
    socket_path: Option<&str>,
) -> anyhow::Result<std::sync::mpsc::Receiver<ChatInput>> {
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_stdin_reader(tx.clone());

    #[cfg(unix)]
    if let Some(path) = socket_path {
        spawn_socket_listener(tx.clone(), path)?;
    }
    #[cfg(not(unix))]
    if socket_path.is_some() {
        anyhow::bail!("--chat-socket is only supported on unix targets");
    }
    drop(tx);
    Ok(rx)
}
