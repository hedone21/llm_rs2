# Benchmark Profile JSON Schema

Each benchmark run generates a JSON file in `results/data/`. This document describes the structure and fields of that JSON file.

## Root Object

| Field | Type | Description |
| :--- | :--- | :--- |
| `version` | `integer` | Schema version number (currently `1`). |
| `metadata` | `object` | static information about the run environment and command. |
| `baseline` | `object` | System state captured before the benchmark started. |
| `benchmark_results` | `object` | Extracted high-level metrics (e.g., TTFT, Tokens/sec). |
| `events` | `array` | List of named timestamped events (e.g., "ModelLoadStart"). |
| `timeseries` | `array` | Detailed time-series data collected during the run. |

## Metadata

| Field | Type | Description |
| :--- | :--- | :--- |
| `date` | `string` | ISO 8601 timestamp of when the benchmark started. |
| `command` | `string` | The full command executed on the device. |
| `model` | `string` | Name of the model used (e.g., `llama3.2-1b`). |
| `backend` | `string` | Backend used (`cpu` or `opencl`). |
| `num_tokens` | `integer` | Number of tokens generated. |
| `prefill_type` | `string` | Identifier for the prompt/prefill used. |
| `cpu_models` | `array` | List of strings describing the CPU cores (e.g., "Little", "Big"). |
| `foreground_app` | `string` | (Optional) Name of the app running in foreground during test. |

## Baseline

| Field | Type | Description |
| :--- | :--- | :--- |
| `avg_memory_used_mb` | `float` | Average system memory used (MB) before the run. |
| `avg_gpu_load_percent` | `float` | Average GPU load (%) before the run. |
| `avg_start_temp_c` | `float` | Average battery temperature (°C) before the run. |

## Benchmark Results

High-level metrics parsed from the standard output.

| Field | Type | Description |
| :--- | :--- | :--- |
| `ttft_ms` | `float` | Time To First Token in milliseconds. |
| `tbt_ms` | `float` | Average Time Between Tokens in milliseconds. |
| `tokens_per_sec` | `float` | Generation speed in tokens per second. |

## Timeseries Data

A list of snapshots taken at regular intervals (default 1.0s).

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `string` | ISO 8601 timestamp of the snapshot. |
| `temp_c` | `float` | Current battery temperature (°C). |
| `mem_used_mb` | `float` | Total system memory usage (MB). |
| `cpu_load_percent` | `float` | Total system CPU load (%). |
| `gpu_load_percent` | `float` | GPU utilization (%). |
| `gpu_freq_hz` | `integer` | GPU clock frequency in Hz. |
| `cpu_freqs_khz` | `array` | List of integers representing current frequency of each CPU core (kHz). |
| `process_mem_mb` | `float` | Memory usage of the target process (MB). |
| `process_cpu_percent` | `float` | CPU usage of the target process (normalized 0-100% of total system). |

## Events

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `string` | Name of the event (e.g., "ModelLoadStart", "PrefillStart"). |
| `timestamp` | `string` | ISO 8601 timestamp when the event occurred. |
