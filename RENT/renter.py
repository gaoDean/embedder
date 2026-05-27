import time
import json
import subprocess
import os

# Configuration
TARGET_RATIO = 210
MAX_TFLOPS = 90
MIN_TFLOPS = 20
TARGET_DISK_SIZE = 200 # GB
CHECK_INTERVAL_SECONDS = 0.1

all_time_greatest = 0

def alert_user(message):
    print(f"\n[ALERT] {message}")
    # Terminal bell
    print('\a')

def check_instances():
    global all_time_greatest


    # Load blacklist
    try:
        with open("blacklist.txt", "r") as f:
            blacklist = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
    except FileNotFoundError:
        blacklist = set()

    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Querying Vast.ai API...")

        # Run the vastai CLI search command
        # The filter 'verified=True' is applied. We request raw JSON output.
        result = subprocess.check_output(
            # ["vastai", "search", "offers", "verified=True", "--raw"],
            ["vastai", "search", "offers", f"num_gpus=1 reliability > 0.99 rented=False verified=False disk_space >= {TARGET_DISK_SIZE}", "--raw"],
            text=True
        )

        offers = json.loads(result)

        best_ratio = 0
        best_offer = None

        for offer in offers:
            machine_id = str(offer.get("machine_id", ""))
            host_id = str(offer.get("host_id", ""))
            if machine_id in blacklist or host_id in blacklist:
                continue

            # total_flops is provided in TFLOPS by Vast.ai
            tflops = offer.get("total_flops", 0)

            # Recalculate price per hour for the specific disk size
            dph_base = offer.get("dph_base", 0)
            storage_cost_per_month = offer.get("storage_cost", 0)
            # Convert monthly per GB cost to hourly cost for the requested disk size (assumes 30 days/month)
            disk_cost_per_hour = (storage_cost_per_month / (30 * 24)) * TARGET_DISK_SIZE
            dph = dph_base + disk_cost_per_hour

            if dph <= 0:
                continue

            ratio = tflops / dph

            # Keep track of best ratio for debugging, only for instances meeting TFLOPS reqs
            if tflops <= MAX_TFLOPS and tflops >= MIN_TFLOPS:
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_offer = offer

            if ratio >= TARGET_RATIO and tflops <= MAX_TFLOPS and tflops >= MIN_TFLOPS:
                machine_id = offer.get('id')
                gpu_name = offer.get('gpu_name', 'Unknown')

                print(f"\n[MATCH FOUND] ID: {machine_id} | GPU: {gpu_name}")
                print(f"TFLOPS: {tflops:.2f} | Price: ${dph:.4f}/hr | Ratio: {ratio:.2f} Tflops/$")

                alert_user(f"Target instance found! Ratio: {ratio:.2f}")

                # --- AUTO RENT EXECUTION ---
                print(f"Attempting to rent instance {machine_id}...")
                rent_result = subprocess.run([
                    "vastai", "create", "instance", str(machine_id),
                    "--disk", str(TARGET_DISK_SIZE),
                    "--template_hash", "2e25192804480e62758bcac2e45ea3fd"
                ], capture_output=True, text=True)
                print(rent_result.stdout)

                return True

        if best_ratio > all_time_greatest:
            all_time_greatest = best_ratio

        print(f"No matching instances found. Best ratio seen this run: {best_ratio:.2f} Tflops/$ (GPU: {best_offer.get('gpu_name') if best_offer else 'None'})")
        if all_time_greatest != 0:
            print(f"Previously best ratio: {all_time_greatest:.2f}")

        return False

    except subprocess.CalledProcessError as e:
        print(f"Error running vastai command. Ensure you are logged in using 'vastai set api-key <KEY>'.")
        print(f"Command output: {e.output}")
        return False
    except Exception as e:
        print(f"Error during checking: {e}")
        return False

def main():
    print("==========================================================")
    print(f" Starting Vast.ai Auto-Renter (Target: >{TARGET_RATIO} TFLOPS/$)")
    print("==========================================================")
    print("IMPORTANT: You must configure your API key first by running:")
    print("  uv run vastai set api-key <YOUR_API_KEY>\n")

    # Do a quick test to see if vastai is working
    try:
        subprocess.check_output(["vastai", "show", "instances-v1"])
    except FileNotFoundError:
        print("CRITICAL: 'vastai' command not found. Ensure you are running this within the 'uv' environment.")
        return
    except subprocess.CalledProcessError:
        print("WARNING: 'vastai show instances-v1' failed. You probably need to set your API key.")

    print(f"Checking every {CHECK_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")

    while True:
        found = check_instances()
        if found:
            print("\nSuccessfully found target instance. Exiting monitor.")
            break

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
