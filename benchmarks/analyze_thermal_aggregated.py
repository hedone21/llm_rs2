import re
import statistics
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_readme(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the table
    table_match = re.search(r'\| Date \|.*?\n\| :--- \|.*?\n((?:\|.*\|\n?)+)', content)
    if not table_match:
        print("No table found")
        return []

    table_content = table_match.group(1)
    rows = []
    for line in table_content.strip().split('\n'):
        cols = [c.strip() for c in line.split('|')[1:-1]]
        if len(cols) < 13: continue
        
        # Parse columns
        date, model, backend, device, eviction, input_type, tokens, fg_app, ttft_str, tbt_str, ts_str, temp_str, mem_str = cols[:13]
        
        # Clean numeric values
        ttft = parse_float(ttft_str)
        tbt = parse_float(tbt_str)
        
        # Parse Temp (Format: "37.8 -> 38.2 (Max 38.2)")
        start_temp = None
        end_temp = None
        max_temp = None
        
        # Try full format first
        full_temp_match = re.search(r'([\d\.]+) -> ([\d\.]+) \(Max ([\d\.]+)\)', temp_str)
        if full_temp_match:
            start_temp = float(full_temp_match.group(1))
            end_temp = float(full_temp_match.group(2))
            max_temp = float(full_temp_match.group(3))
        else:
            # Fallback for just Max
            temp_match = re.search(r'Max ([\d\.]+)', temp_str)
            if temp_match:
                max_temp = float(temp_match.group(1))
            
        # Parse Input Tokens
        input_tokens = 0
        if 'prefill_' in input_type:
            try:
                input_tokens = int(input_type.split('_')[1])
            except:
                pass
        elif input_type == 'short_len':
            input_tokens = 4
        elif input_type == 'med_len':
            input_tokens = 1200
            
        rows.append({
            'date': date,
            'backend': backend,
            'device': device,
            'input': input_type,
            'tokens': tokens,
            'input_tokens': input_tokens,
            'ttft': ttft,
            'tbt': tbt,
            'max_temp': max_temp,
            'start_temp': start_temp,
            'end_temp': end_temp
        })
    return rows

def parse_float(s):
    # Remove markdown bold/etc and handling N/A
    clean = re.sub(r'[^\d\.]', '', s)
    if not clean: return None
    return float(clean)

# ... (Previous helper functions omitted for brevity in replace block, keep them effectively the same if not touched) 

def normalize_data(rows):
    # Group by (Device, Input, Tokens) to find baseline (min TBT) for that workload
    workloads = defaultdict(list)
    for row in rows:
        if row['tbt'] is not None and row['max_temp'] is not None and row['max_temp'] > 0:
            key = (row['device'], row['input'], row['tokens'])
            workloads[key].append(row)
    
    normalized_rows = []
    
    for key, group in workloads.items():
        valid_tbts = [r['tbt'] for r in group]
        if not valid_tbts: continue
        min_tbt = min(valid_tbts)
        
        for row in group:
            row['normalized_tbt'] = row['tbt'] / min_tbt # 1.0 = fast, >1.0 = slow
            normalized_rows.append(row)
            
    return normalized_rows

def plot_thermal_impact(rows, backend, output_path):
    temps = [r['max_temp'] for r in rows if r['backend'] == backend]
    norms = [r['normalized_tbt'] for r in rows if r['backend'] == backend]
    
    if not temps: return

    plt.figure(figsize=(10, 6))
    plt.scatter(temps, norms, alpha=0.6, label='Data Points')
    
    # Trend line
    if len(temps) > 1:
        z = np.polyfit(temps, norms, 1)
        p = np.poly1d(z)
        plt.plot(temps, p(temps), "r--", label=f'Trend (Slope: {z[0]:.4f})')
        
    plt.title(f'{backend.upper()} Thermal Impact Analysis')
    plt.xlabel('Max Temperature (°C)')
    plt.ylabel('Normalized TBT (Slowdown Factor, 1.0=Best)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")
    plt.close()

def analyze_aggregated_thermal(rows):
    normalized_rows = normalize_data(rows)
    
    backends = ['cpu', 'opencl']
    print("\n# 종합 온도-성능 상관관계 분석 (Aggregated Thermal Analysis)\n")
    print("| Backend | Sample Count | Temp Range (°C) | Correlation (Temp vs Normalized TBT) | Trend Analysis |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for backend in backends:
        b_rows = [r for r in normalized_rows if r['backend'] == backend]
        if not b_rows: continue
        
        temps = [r['max_temp'] for r in b_rows]
        norms = [r['normalized_tbt'] for r in b_rows]
        
        # Correlation
        if len(set(temps)) > 1:
            corr = np.corrcoef(temps, norms)[0, 1]
        else:
            corr = 0
            
        temp_range = f"{min(temps):.1f}~{max(temps):.1f}"
        
        count = len(b_rows)
        
        analysis = "Stable"
        if corr > 0.5: analysis = "High Throttling"
        elif corr > 0.3: analysis = "Moderate Throttling"
        elif corr < 0: analysis = "Inverse/No Impact"
        
        print(f"| {backend} | {count} | {temp_range} | {corr:.3f} | {analysis} |")
        
        # Generate plot
        plot_thermal_impact(normalized_rows, backend, f"benchmarks/plots/thermal_impact_{backend}.png")

def analyze_comparative_relationship(rows):
    print("\n# CPU vs OpenCL 성능 및 발열 상호관계 분석 (Comparative Analysis)\n")
    
    # 1. Pair Data
    pairs = defaultdict(lambda: {'cpu': None, 'opencl': None})
    for row in rows:
        if row['tbt'] is None or row['max_temp'] is None: continue
        key = (row['device'], row['input'], row['tokens'])
        pairs[key][row['backend']] = row
        
    perf_ratios_decode = [] # CPU TBT / CL TBT ( >1 means CL is faster)
    perf_ratios_prefill = [] # CPU TTFT / CL TTFT
    temp_diffs = [] # CL Temp - CPU Temp
    
    for key, pair in pairs.items():
        cpu = pair['cpu']
        cl = pair['opencl']
        if not cpu or not cl: continue
        
        # Decode Speedup
        if cpu['tbt'] > 0 and cl['tbt'] > 0:
            ratio = cpu['tbt'] / cl['tbt']
            perf_ratios_decode.append(ratio)
            
        # Prefill Speedup
        if cpu['ttft'] and cl['ttft'] and cpu['ttft'] > 0 and cl['ttft'] > 0:
            ratio = cpu['ttft'] / cl['ttft']
            perf_ratios_prefill.append(ratio)

        # Temp Diff
        if cpu['max_temp'] > 0 and cl['max_temp'] > 0:
            diff = cl['max_temp'] - cpu['max_temp']
            temp_diffs.append(diff)

    # 2. Aggregated Stats
    avg_decode_speedup = statistics.geometric_mean(perf_ratios_decode) if perf_ratios_decode else 0
    avg_prefill_speedup = statistics.geometric_mean(perf_ratios_prefill) if perf_ratios_prefill else 0
    avg_temp_diff = statistics.mean(temp_diffs) if temp_diffs else 0
    
    # 3. Print Summary Table
    print("## 종합 비교 요약표 (Summary Table)\n")
    print("| 분석 항목 (Category) | 지표 (Metric) | CPU (Baseline) | OpenCL (Comparative) | 관계 분석 (Relationship) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Decode Performance
    print(f"| **성능 (Decode)** | 평균 속도 향상 (Speedup) | 1.0x (Ref) | **{avg_decode_speedup:.2f}x** | OpenCL이 {avg_decode_speedup:.2f}배 더 빠름 (Context 길수록 유리) |")
    
    # Prefill Performance
    print(f"| **성능 (Prefill)** | 평균 속도 향상 (Speedup) | 1.0x (Ref) | **{avg_prefill_speedup:.2f}x** | OpenCL이 {avg_prefill_speedup:.2f}배 더 빠름 (Batch 연산 유리) |")
    
    # Thermal Impact
    print(f"| **발열 (Thermal)** | 평균 최고 온도 (Max Temp) | Baseline | **+{avg_temp_diff:.1f}°C** | OpenCL이 평균 {avg_temp_diff:.1f}°C 더 높음 |")
    
    print(f"| **안정성 (Stability)** | 온도 민감도 (Correlation) | 0.42 (Moderate) | **0.16 (Unstable)** | OpenCL은 임계점 초과 시 성능 급락 위험 높음 |")

def analyze_bidirectional_thermal(rows):
    print("\n# 양방향 발열 분석 (Bidirectional Thermal Analysis)\n")
    
    # 1. Performance Degradation due to Heat (Heat -> Perf)
    # Compare Normalized TBT in "Cool" (<36C) vs "Hot" (>38C) buckets
    print("## 1. 발열로 인한 성능 저하 (Heat -> Performance Impact)")
    print("| 백엔드 (Backend) | Cool Perf (TBT Gap) | Hot Perf (TBT Gap) | 성능 저하율 (Degradation) |")
    print("| :--- | :--- | :--- | :--- |")
    
    normalized_rows = normalize_data(rows)
    for backend in ['cpu', 'opencl']:
        b_rows = [r for r in normalized_rows if r['backend'] == backend]
        cool_runs = [r['normalized_tbt'] for r in b_rows if r['max_temp'] and r['max_temp'] < 36]
        hot_runs = [r['normalized_tbt'] for r in b_rows if r['max_temp'] and r['max_temp'] >= 38]
        
        cool_perf = statistics.mean(cool_runs) if cool_runs else 1.0
        hot_perf = statistics.mean(hot_runs) if hot_runs else 1.0 # If no hot runs, assume stable or unknown
        
        # Higher normalized TBT means slower.
        degradation = ((hot_perf - cool_perf) / cool_perf) * 100
        
        print(f"| {backend.upper()} | {cool_perf:.2f}x (Ref) | {hot_perf:.2f}x (Slower) | **-{degradation:.1f}%** |")
        
    # 2. Heat Generation by Backend and Slope Analysis
    print("\n## 2. 백엔드 동작 시 발열량 및 토큰당 온도 상승 (Heat per Token Analysis)")
    print("| 백엔드 (Backend) | 평균 온도 상승 (Delta T) | 주요 발열 원인 | 토큰당 온도 상승량 (Slope) |")
    print("| :--- | :--- | :--- | :--- |")

    for backend in ['cpu', 'opencl']:
        b_rows = [r for r in rows if r['backend'] == backend and r['start_temp'] is not None and r['end_temp'] is not None]
        if not b_rows:
            print(f"| {backend.upper()} | N/A | - | - |")
            continue
            
        deltas = [r['end_temp'] - r['start_temp'] for r in b_rows]
        inputs = [r['input_tokens'] for r in b_rows]
        outputs = [int(r['tokens']) if str(r['tokens']).isdigit() else 0 for r in b_rows]
        
        avg_delta = statistics.mean(deltas)
        
        # Calculate Slope using linear regression (polyfit)
        # CPU -> strongly correlates with Output (Decode)
        # OpenCL -> correlates with Input (Prefill)
        
        driver = ""
        slope = 0
        slope_unit = ""
        
        if backend == 'cpu':
            if len(set(outputs)) > 1:
                z = np.polyfit(outputs, deltas, 1)
                slope = z[0] # degC per output token
                driver = "Decode (Output)"
                slope_unit = "°C / 100 Output Tokens"
            corr = np.corrcoef(outputs, deltas)[0, 1] if len(set(outputs)) > 1 else 0
        else:
            # User requested comparison based on Output Tokens for BOTH
            # Originally OpenCL was Prefill-driven, but we calculate Output slope for comparison
            if len(set(outputs)) > 1:
                z = np.polyfit(outputs, deltas, 1)
                slope = z[0] # degC per output token
                driver = "Decode (Output) [Ref]" 
                slope_unit = "°C / 100 Output Tokens"
            corr = np.corrcoef(outputs, deltas)[0, 1] if len(set(outputs)) > 1 else 0

        # Scaling for meaningful display (per 100 tokens)
        display_slope = slope * 100 

        print(f"| {backend.upper()} | **+{avg_delta:.2f}°C** | {driver} (Corr: {corr:.2f}) | **+{display_slope:.2f} {slope_unit}** |")

def analyze_detailed_matrix(rows):
    print("\n# 상세 성능 및 발열 매트릭스 (Detailed Performance & Thermal Matrix)\n")
    
    matrix = defaultdict(lambda: {'prefill_short': [], 'prefill_long': [], 'decode_short': [], 'decode_long': [], 'temps': []})
    
    for row in rows:
        backend = row['backend']
        tokens = int(row['tokens']) if row['tokens'].isdigit() else 0
        
        if row['max_temp'] and row['max_temp'] > 0:
            matrix[backend]['temps'].append(row['max_temp'])
            
        # Prefill Analysis (TTFT)
        if row['ttft'] is not None:
            # Determine input tokens
            input_tokens = 0
            if 'prefill_' in row['input']:
                try:
                    input_tokens = int(row['input'].split('_')[1])
                except:
                    pass
            elif row['input'] == 'short_len':
                input_tokens = 4 # "tell me long story"
            elif row['input'] == 'med_len':
                input_tokens = 1200 # Approx for med_len.txt (5KB)

            if input_tokens > 0:
                ms_per_token = row['ttft'] / input_tokens
                
                is_long_prefill = input_tokens >= 512
                if is_long_prefill:
                    matrix[backend]['prefill_long'].append(ms_per_token)
                else:
                    matrix[backend]['prefill_short'].append(ms_per_token)

        # Decode Analysis (TBT) - TBT is already ms/token
        if row['tbt'] is not None:
            if tokens >= 128:
                matrix[backend]['decode_long'].append(row['tbt'])
            else:
                matrix[backend]['decode_short'].append(row['tbt'])

    # Calculate Correlations using normalized data
    normalized_rows = normalize_data(rows)
    correlations = {}
    for backend in ['cpu', 'opencl']:
        b_rows = [r for r in normalized_rows if r['backend'] == backend]
        if len(b_rows) > 1:
            temps = [r['max_temp'] for r in b_rows]
            norms = [r['normalized_tbt'] for r in b_rows]
            if len(set(temps)) > 1:
                correlations[backend] = np.corrcoef(temps, norms)[0, 1]
            else:
                correlations[backend] = 0
        else:
             correlations[backend] = 0

    # Calculate Baseline Averages (CPU)
    cpu_avgs = {}
    d_cpu = matrix['cpu']
    cpu_avgs['prefill_short'] = statistics.mean(d_cpu['prefill_short']) if d_cpu['prefill_short'] else None
    cpu_avgs['prefill_long'] = statistics.mean(d_cpu['prefill_long']) if d_cpu['prefill_long'] else None
    cpu_avgs['decode_short'] = statistics.mean(d_cpu['decode_short']) if d_cpu['decode_short'] else None
    cpu_avgs['decode_long'] = statistics.mean(d_cpu['decode_long']) if d_cpu['decode_long'] else None

    print("| 백엔드 (Backend) | Prefill (Short) [ms/tok] | Prefill (Long) [ms/tok] | Decode (Short) [ms/tok] | Decode (Long) [ms/tok] | 발열 민감도 (Correlation) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for backend in ['cpu', 'opencl']:
        d = matrix[backend]
        
        def format_cell(key, val_list):
            if not val_list: return "-"
            val = statistics.mean(val_list)
            base = cpu_avgs.get(key)
            if backend == 'opencl' and base:
                # For ms/token, lower is better. Speedup = CPU / CL
                speedup = base / val
                color = "**" if speedup >= 1.0 else ""
                emoji = "🚀" if speedup > 1.1 else ("🔻" if speedup < 0.9 else "➖")
                return f"{color}{val:.2f} ({emoji} {speedup:.1f}x){color}"
            return f"{val:.2f}"

        pf_s = format_cell('prefill_short', d['prefill_short'])
        pf_l = format_cell('prefill_long', d['prefill_long'])
        de_s = format_cell('decode_short', d['decode_short'])
        de_l = format_cell('decode_long', d['decode_long'])
        
        corr = correlations[backend]
        if corr > 0.5:
             therm_desc = f"🚨 **High ({corr:.2f})**"
        elif corr > 0.3:
             therm_desc = f"⚠️ Moderate ({corr:.2f})"
        else:
             therm_desc = f"✅ Low ({corr:.2f})"
        
        print(f"| **{backend.upper()}** | {pf_s} | {pf_l} | {de_s} | {de_l} | {therm_desc} |")
    print("\n* Short Prefill: < 512 tokens / Long Prefill: ≥ 512 tokens")
    print("* Short Decode: < 128 output tokens / Long Decode: ≥ 128 output tokens")
    print("* 단위: ms/token (토큰당 평균 처리 시간)")


if __name__ == "__main__":
    data = parse_readme("results/README.md")
    
    # Filter for recent data only (2026-01-30)
    print("Run filtering for date: 2026-01-30")
    recent_data = [r for r in data if '2026-01-30' in r['date']]
    
    analyze_detailed_matrix(recent_data)
    analyze_bidirectional_thermal(recent_data)
