#!/usr/bin/env python3
import argparse
import subprocess
import time
import sys
import threading

def run_adb_command(cmd):
    """Runs an adb command and returns the output."""
    full_cmd = f"adb {cmd}"
    try:
        result = subprocess.run(full_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running adb command '{cmd}': {e.stderr}")
        return None

def switch_apps(apps, interval, stop_event):
    """Periodically switches between provided apps/intents to simulate load."""
    if not apps:
        print("[Stress] No background apps provided to switch. Running only main command.")
        return

    print(f"[Stress] Starting background app switching every {interval}s with apps: {apps}")
    
    idx = 0
    while not stop_event.is_set():
        app = apps[idx % len(apps)]
        
        # Check if it looks like a URL (http/https) or pure package name
        if app.startswith("http://") or app.startswith("https://"):
            # Launch URL (handled by default browser or app like YouTube)
            cmd = f"shell am start -a android.intent.action.VIEW -d \"{app}\""
        elif "/" in app:
             # Component name format: package/activity
            cmd = f"shell am start -n {app}"
        else:
            # Just package name
            cmd = f"shell monkey -p {app} -c android.intent.category.LAUNCHER 1"

        run_adb_command(cmd)
        
        idx += 1
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Run a command on Android in the background while switching apps to induce stress.")
    parser.add_argument("--cmd", required=True, help="The command to run on the device (absolute path recommended).")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run the test in seconds.")
    parser.add_argument("--switch-interval", type=int, default=10, help="Interval in seconds to switch foreground apps.")
    parser.add_argument("--background-apps", default="com.android.settings", 
                        help="Comma-separated list of package names or URLs to switch between. Default: com.android.settings")
    
    args = parser.parse_args()

    # Parse apps list
    apps_list = [x.strip() for x in args.background_apps.split(",") if x.strip()]

    print(f"[Stress] Preparing to run: {args.cmd}")
    print(f"[Stress] Duration: {args.duration}s, Switch Interval: {args.switch_interval}s")

    # 0. Wake up screen
    print("[Stress] Waking up device...")
    run_adb_command("shell input keyevent KEYCODE_WAKEUP")
    run_adb_command("shell input keyevent KEYCODE_MENU") # Unlock if no password

    # 1. Start the main workload in the background
    bg_cmd = f"shell \"nohup {args.cmd} > /data/local/tmp/stress_output.log 2>&1 & echo \$!\""
    pid = run_adb_command(bg_cmd)
    
    if not pid:
        print("Failed to start background process.")
        sys.exit(1)
        
    print(f"[Stress] Background process started with PID: {pid}")

    # 2. Start App Switching Thread
    stop_event = threading.Event()
    switcher_thread = threading.Thread(target=switch_apps, args=(apps_list, args.switch_interval, stop_event))
    switcher_thread.start()

    # 3. Monitor
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            # Check if process is still running
            check_cmd = f"shell \"ps -p {pid}\""
            res = run_adb_command(check_cmd)
            if pid not in res:
                print(f"[Stress] Process {pid} finished early or crashed.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("[Stress] Interrupted by user.")

    # 4. Cleanup
    stop_event.set()
    switcher_thread.join()
    
    print("[Stress] Stopping background process...")
    run_adb_command(f"shell kill {pid}")
    
    # Check output
    print(f"[Stress] processing output log size...")
    run_adb_command(f"shell ls -lh /data/local/tmp/stress_output.log")
    print("[Stress] Done. You can check /data/local/tmp/stress_output.log on device.")

if __name__ == "__main__":
    main()
