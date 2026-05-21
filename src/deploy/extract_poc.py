import os
import glob
import pandas as pd
import numpy as np
import argparse
import shutil
from tqdm import tqdm

# Helper function to format numbers with spaces
def format_num(n):
    if isinstance(n, float):
        return f"{n:,.2f}".replace(",", " ")
    return f"{n:,}".replace(",", " ")

def find_peak_window_and_extract(raw_dir, out_dir, duration_sec, num_cars):
    print(f"Scanning original dataset in {raw_dir}...")
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    
    if not csv_files:
        print("No CSV files found in the raw directory.")
        return
        
    print(f"Found {len(csv_files)} files. Extracting temporal bounds to find the network peak...")
    
    car_spans = []
    for f in tqdm(csv_files, desc="Reading bounds"):
        try:
            # We only read the 'Time' column to save RAM/Time
            df_time = pd.read_csv(f, usecols=['Time'])
            if len(df_time) > 0:
                start_t = df_time['Time'].iloc[0]
                end_t = df_time['Time'].iloc[-1]
                car_spans.append({'file': f, 'start': start_t, 'end': end_t})
        except Exception as e:
            continue
            
    spans = pd.DataFrame(car_spans)
    
    global_start = spans['start'].min()
    global_end = spans['end'].max()
    
    print(f"\nGlobal Simulation Time: {global_start:.1f}s to {global_end:.1f}s")
    
    # Sliding window to find the peak (step by 50 seconds for fine granularity)
    best_t = global_start
    max_active_cars = 0
    
    print(f"Searching for the busiest {duration_sec}s time window...")
    for t in np.arange(global_start, global_end - duration_sec + 1, 50):
        t_end = t + duration_sec
        # A car is active in this window if it starts before the window ends and ends after the window starts
        active = spans[(spans['start'] <= t_end) & (spans['end'] >= t)]
        if len(active) > max_active_cars:
            max_active_cars = len(active)
            best_t = t
            
    best_t_end = best_t + duration_sec
    print(f"\n✅ Peak Network Window Found: {best_t:.1f}s to {best_t_end:.1f}s")
    print(f"Maximum concurrent cars in this window: {max_active_cars}")
    
    # Select cars overlapping with the peak window
    overlaps = spans[(spans['start'] <= best_t_end) & (spans['end'] >= best_t)].copy()
    
    # Calculate how much of the car's lifespan is actually inside the window
    overlaps['overlap_duration'] = np.minimum(overlaps['end'], best_t_end) - np.maximum(overlaps['start'], best_t)
    
    # Sort by the longest presence in the window to get the richest trajectories
    overlaps = overlaps.sort_values('overlap_duration', ascending=False)
    
    if num_cars > 0:
        selected_cars = overlaps.head(num_cars)
        print(f"Selecting the top {len(selected_cars)} cars with the longest presence in the peak window.")
    else:
        selected_cars = overlaps
        print(f"Selecting all {len(selected_cars)} cars present in the peak window.")
        
    # Create Output Directory
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nExtracting and truncating trajectories to {out_dir}...")
    for _, row in tqdm(selected_cars.iterrows(), total=len(selected_cars), desc="Extracting"):
        df = pd.read_csv(row['file'])
        # Truncate the dataframe strictly to the time window
        df_filtered = df[(df['Time'] >= best_t) & (df['Time'] <= best_t_end)]
        
        if len(df_filtered) > 0:
            out_file = os.path.join(out_dir, os.path.basename(row['file']))
            df_filtered.to_csv(out_file, index=False)
            
    print("Extraction complete!\n")
    return out_dir

def run_analytics(data_dir, report_dir):
    print("==================================================")
    print("Running Global Analytics on Proof-of-Concept Dataset")
    print("==================================================")
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    total_files = len(csv_files)
    total_rows = 0
    total_size_bytes = 0
    rows_per_file = []
    lifespans = []
    start_times = []
    end_times = []

    # --- New Network Metric Trackers ---
    global_delay_sum = 0
    global_delay_count = 0
    global_plr_sum = 0
    global_plr_count = 0

    print(f"Found {total_files} CSV files. Processing...")

    for file in tqdm(csv_files, desc="Analyzing"):
        try:
            # Get file size
            total_size_bytes += os.path.getsize(file)
            
            # Read the csv file
            df = pd.read_csv(file)
            num_rows = len(df)
            rows_per_file.append(num_rows)
            total_rows += num_rows
            
            # Extract Global Networking Metrics
            if 'AvgMsgDelay' in df.columns:
                global_delay_sum += df['AvgMsgDelay'].sum()
                global_delay_count += df['AvgMsgDelay'].count()
            if 'PacketLossRate' in df.columns:
                global_plr_sum += df['PacketLossRate'].sum()
                global_plr_count += df['PacketLossRate'].count()
            
            # Try to identify a time column to calculate lifespan and simulation duration
            time_col = None
            for col in df.columns:
                if col.lower() in ['time', 't', 'timestamp', 'simtime']:
                    time_col = col
                    break
            
            if time_col and num_rows > 0:
                start_t = df[time_col].iloc[0]
                end_t = df[time_col].iloc[-1]
                lifespans.append(end_t - start_t)
                start_times.append(start_t)
                end_times.append(end_t)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Calculate file sizes
    total_size_gb = total_size_bytes / (1024 ** 3)
    avg_size_mb = (total_size_bytes / total_files) / (1024 ** 2) if total_files > 0 else 0

    # Calculate Global Network Averages
    global_avg_delay = (global_delay_sum / global_delay_count) if global_delay_count > 0 else 0
    global_avg_plr = (global_plr_sum / global_plr_count) if global_plr_count > 0 else 0

    # --- Collect Results into a DataFrame ---
    results_data = {
        "Metric": [
            "Total CSV files (cars)",
            "Total rows across all files",
            "Total dataset size",
            "Average file size"
        ],
        "Value": [
            f"{format_num(total_files)} files",
            f"{format_num(total_rows)} rows",
            f"{format_num(round(total_size_gb, 2))} GB",
            f"{format_num(round(avg_size_mb, 2))} MB"
        ]
    }

    if rows_per_file:
        results_data["Metric"].extend([
            "Average rows per file",
            "Highest number of rows",
            "Lowest number of rows"
        ])
        results_data["Value"].extend([
            f"{format_num(int(round(np.mean(rows_per_file))))} rows",
            f"{format_num(np.max(rows_per_file))} rows",
            f"{format_num(np.min(rows_per_file))} rows"
        ])
        
    if global_delay_count > 0:
        results_data["Metric"].extend([
            "Global Avg Message Delay",
            "Global Packet Loss Rate"
        ])
        results_data["Value"].extend([
            f"{global_avg_delay:.6f} seconds ({global_avg_delay * 1000:.2f} ms)",
            f"{global_avg_plr * 100:.2f} %"
        ])

    avg_lifespan_sec = 0
    avg_lifespan_steps = 0
    if lifespans:
        # The time column in the CSV is already in seconds (e.g., 17002.1)
        avg_lifespan_sec = np.mean(lifespans)
        avg_lifespan_steps = avg_lifespan_sec * 10
        results_data["Metric"].extend([
            "Average lifespan"
        ])
        results_data["Value"].extend([
            f"{format_num(round(avg_lifespan_sec, 2))} seconds ({format_num(int(round(avg_lifespan_steps)))} steps)"
        ])
        
    duration_sec = 0
    duration_steps = 0
    if start_times and end_times:
        sim_start = min(start_times)
        sim_end = max(end_times)
        # The time difference is in seconds
        duration_sec = sim_end - sim_start
        duration_steps = duration_sec * 10
        results_data["Metric"].extend([
            "Simulation start time",
            "Simulation end time",
            "Total simulation duration"
        ])
        results_data["Value"].extend([
            f"{format_num(round(sim_start, 2))} s",
            f"{format_num(round(sim_end, 2))} s",
            f"{format_num(round(duration_sec, 2))} seconds ({format_num(int(round(duration_steps)))} steps)"
        ])

    results_df = pd.DataFrame(results_data)
    
    # Print out nicely to console
    print("\n--- ANALYTICS RESULTS ---")
    for _, row in results_df.iterrows():
        print(f"{row['Metric']:<30}: {row['Value']}")
    print("-------------------------\n")

    # --- Build and Save Markdown Report ---
    report_lines = [
        "# Proof-of-Concept Dataset Analysis Report\n",
        "## General Information",
        f"- **Total CSV files (cars):** {format_num(total_files)} files",
        f"- **Total rows across all files:** {format_num(total_rows)} rows",
        f"- **Total dataset size:** {format_num(round(total_size_gb, 2))} GB",
        f"- **Average file size:** {format_num(round(avg_size_mb, 2))} MB"
    ]

    if rows_per_file:
        report_lines.extend([
            f"- **Average rows per file:** {format_num(int(round(np.mean(rows_per_file))))} rows",
            f"- **Highest number of rows:** {format_num(np.max(rows_per_file))} rows",
            f"- **Lowest number of rows:** {format_num(np.min(rows_per_file))} rows"
        ])
        
    if global_delay_count > 0:
        report_lines.extend([
            "\n## Global Network Health",
            f"- **Global Average Message Delay:** {global_avg_delay:.6f} seconds ({global_avg_delay * 1000:.2f} ms)",
            f"- **Global Average Packet Loss Rate:** {global_avg_plr * 100:.2f} %"
        ])

    report_lines.extend(["\n## Temporal Information"])

    if lifespans:
        report_lines.append(f"- **Average lifespan of a car:** {format_num(round(avg_lifespan_sec, 2))} seconds ({format_num(int(round(avg_lifespan_steps)))} steps)")
        
    if start_times and end_times:
        report_lines.append(f"- **Simulation start time:** {format_num(round(sim_start, 2))} s")
        report_lines.append(f"- **Simulation end time:** {format_num(round(sim_end, 2))} s")
        report_lines.append(f"- **Total simulation duration:** {format_num(round(duration_sec, 2))} seconds ({format_num(int(round(duration_steps)))} steps)")
    else:
        report_lines.append("- *(Could not find a recognized time column to calculate durations)*")

    report_md = "\n".join(report_lines)

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "poc_dataset_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"\nReport successfully saved to {os.path.abspath(report_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a PoC subset of the data based on peak network conditions.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to the original raw dataset folder.")
    parser.add_argument("--out_dir", type=str, default="resources/poc_dataset/raw", help="Path to save the extracted PoC dataset.")
    parser.add_argument("--duration", type=int, default=1000, help="Duration of the time window to extract in seconds.")
    parser.add_argument("--num_cars", type=int, default=1000, help="Number of cars to select inside the peak window. Use -1 for all cars.")
    
    args = parser.parse_args()
    
    find_peak_window_and_extract(args.raw_dir, args.out_dir, args.duration, args.num_cars)
    
    # Run the user's analytics over the new output directory
    run_analytics(args.out_dir, os.path.dirname(args.out_dir))
