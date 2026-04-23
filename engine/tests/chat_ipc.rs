//! Integration tests for the chat IPC input multiplexer. Exercises the
//! Unix socket path end-to-end: bind → connect → send line → verify the
//! mpsc receiver yields the message → stream bytes back → verify the
//! client reads them up to the 0x04 end-of-turn delimiter.

#![cfg(unix)]

use llm_rs2::core::chat_ipc::{
    ChatInput, finish_reply_stream, spawn_socket_listener, write_reply_bytes,
};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

fn unique_sock_path(tag: &str) -> String {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/llm_rs2_chat_ipc_{tag}_{pid}_{nanos}.sock")
}

#[test]
fn socket_accepts_connection_and_delivers_line() {
    let path = unique_sock_path("single");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    // Give the listener thread a moment to bind.
    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(std::path::Path::new(&path).exists(), "socket not bound");

    let mut client = UnixStream::connect(&path).expect("connect");
    client.write_all(b"hello world\n").unwrap();
    client.flush().unwrap();

    let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
    match msg {
        ChatInput::Line(line, writer) => {
            assert_eq!(line, "hello world");
            let writer = writer.expect("socket input must carry a reply writer");
            write_reply_bytes(Some(&writer), b"reply-payload");
            finish_reply_stream(Some(&writer));
        }
        _ => panic!("expected Line"),
    }

    let mut buf = Vec::new();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    // Read until EOT (0x04). finish_reply_stream does NOT shut down the
    // write side, so we stop on the delimiter rather than EOF.
    let mut byte = [0u8; 1];
    while client.read_exact(&mut byte).is_ok() {
        if byte[0] == 0x04 {
            break;
        }
        buf.push(byte[0]);
    }
    assert_eq!(&buf, b"reply-payload");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn socket_supports_multiple_turns_on_same_connection() {
    let path = unique_sock_path("multi");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut client = UnixStream::connect(&path).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    for i in 0..3 {
        let user_line = format!("turn-{i}\n");
        client.write_all(user_line.as_bytes()).unwrap();
        client.flush().unwrap();

        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        let reply = match msg {
            ChatInput::Line(line, writer) => {
                assert_eq!(line, format!("turn-{i}"));
                let writer = writer.expect("writer");
                let payload = format!("ack-{i}");
                write_reply_bytes(Some(&writer), payload.as_bytes());
                finish_reply_stream(Some(&writer));
                payload
            }
            _ => panic!("expected Line"),
        };

        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while client.read_exact(&mut byte).is_ok() {
            if byte[0] == 0x04 {
                break;
            }
            buf.push(byte[0]);
        }
        assert_eq!(String::from_utf8(buf).unwrap(), reply);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn socket_handles_concurrent_connections() {
    let path = unique_sock_path("concurrent");
    let (tx, rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn");

    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mut clients: Vec<UnixStream> = (0..3)
        .map(|_| UnixStream::connect(&path).expect("connect"))
        .collect();
    for (i, c) in clients.iter_mut().enumerate() {
        c.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        writeln!(c, "client-{i}").unwrap();
        c.flush().unwrap();
    }

    // Collect 3 messages regardless of ordering and ACK each one through
    // its own writer.
    let mut lines = Vec::new();
    for _ in 0..3 {
        let msg = rx.recv_timeout(Duration::from_secs(2)).expect("recv");
        if let ChatInput::Line(line, Some(writer)) = msg {
            let reply = format!("ack:{line}");
            write_reply_bytes(Some(&writer), reply.as_bytes());
            finish_reply_stream(Some(&writer));
            lines.push(line);
        } else {
            panic!("expected Line with writer");
        }
    }
    lines.sort();
    assert_eq!(lines, vec!["client-0", "client-1", "client-2"]);

    // Each client should see exactly its own ack payload.
    for (i, c) in clients.iter_mut().enumerate() {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while c.read_exact(&mut byte).is_ok() {
            if byte[0] == 0x04 {
                break;
            }
            buf.push(byte[0]);
        }
        assert_eq!(String::from_utf8(buf).unwrap(), format!("ack:client-{i}"));
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn rebind_removes_stale_socket_file() {
    let path = unique_sock_path("stale");
    // Create a stale regular file at the socket path.
    std::fs::write(&path, b"stale").unwrap();
    assert!(std::path::Path::new(&path).exists());

    let (tx, _rx) = std::sync::mpsc::channel::<ChatInput>();
    spawn_socket_listener(tx, &path).expect("spawn must remove stale file");
    for _ in 0..50 {
        if std::path::Path::new(&path).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // After spawn, the file exists as a socket (bind created it).
    let meta = std::fs::metadata(&path).unwrap();
    assert!(
        std::os::unix::fs::FileTypeExt::is_socket(&meta.file_type()),
        "expected unix socket, got {:?}",
        meta.file_type()
    );

    let _ = std::fs::remove_file(&path);
}
