---
description: Build the generate binary, push to Android device, and run a quick inference test.
---

// turbo-all

1. Build for Android (Release)
   cargo build --target aarch64-linux-android --release --bin generate

2. Push binary to device
   adb push target/aarch64-linux-android/release/generate /data/local/tmp/llm_rs2/generate
   adb shell chmod +x /data/local/tmp/llm_rs2/generate

3. Run sanity test on device
   adb shell "/data/local/tmp/llm_rs2/generate --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b --prompt 'Hello world' -n 128"
