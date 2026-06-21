import os
import glob
import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm

def format_num(n):
    if isinstance(n, float) or isinstance(n, int):
        return f"{n:,.2f}".replace(",", " ") if isinstance(n, float) else f"{n:,}".replace(",", " ")
    return str(n)

def get_sliced_dataset_metrics(data_dir, desc_name, start_t, end_t, target_basenames=None):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if target_basenames is not None:
        csv_files = [f for f in csv_files if os.path.basename(f) in target_basenames]
        
    processed_basenames = set()
    total_files = 0
    total_rows = 0
    
    global_delay_sum = 0
    global_delay_count = 0
    global_plr_sum = 0
    global_plr_count = 0
    
    for file in tqdm(csv_files, desc=f"Analyzing {desc_name}"):
        try:
            df = pd.read_csv(file)
            
            # SLICE: only look at the data within the exact peak window to prevent data leakage
            if 'Time' in df.columns:
                df = df[(df['Time'] >= start_t) & (df['Time'] <= end_t)]
            elif 'simtime' in df.columns:
                df = df[(df['simtime'] >= start_t) & (df['simtime'] <= end_t)]
            elif 't' in df.columns:
                df = df[(df['t'] >= start_t) & (df['t'] <= end_t)]
                
            if len(df) == 0:
                continue # Car wasn't active in this window
                
            processed_basenames.add(os.path.basename(file))
            total_files += 1
            total_rows += len(df)
            
            if 'AvgMsgDelay' in df.columns:
                global_delay_sum += df['AvgMsgDelay'].sum()
                global_delay_count += df['AvgMsgDelay'].count()
            if 'PacketLossRate' in df.columns:
                global_plr_sum += df['PacketLossRate'].sum()
                global_plr_count += df['PacketLossRate'].count()
                
        except Exception as e:
            continue
            
    avg_delay = (global_delay_sum / global_delay_count) if global_delay_count > 0 else 0
    avg_plr = (global_plr_sum / global_plr_count) if global_plr_count > 0 else 0
    
    metrics = {
        "Total Active Vehicles in Window": format_num(total_files),
        "Total Trajectory Steps (Rows)": format_num(total_rows),
        "Avg Message Delay (ms)": f"{avg_delay * 1000:.2f} ms",
        "Average Packet Loss Rate (%)": f"{avg_plr * 100:.2f} %"
    }
    
    return metrics, processed_basenames

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline PoC vs Optimized OMNeT++ Simulation")
    parser.add_argument("--config", type=str, default="resources/poc_dataset/poc_config.json", help="Path to poc_config.json")
    parser.add_argument("--optimized_dir", type=str, required=True, help="Path to the new OMNeT++ dataset generated with Transformer decisions")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
        
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    baseline_dir = config.get("out_dir") # The PoC dataset extracted initially
    start_t = config.get("best_t_start")
    end_t = config.get("best_t_end")
    
    if not baseline_dir or start_t is None or end_t is None:
        print("Error: Invalid config file. Missing out_dir or time window variables.")
        return
        
    if not os.path.exists(baseline_dir):
        print(f"Error: Baseline PoC Dir not found: {baseline_dir}")
        return
        
    if not os.path.exists(args.optimized_dir):
        print(f"Error: Optimized dataset not found: {args.optimized_dir}")
        return
        
    print("=========================================================================")
    print(f"Running Strict Temporal Comparison [{start_t}s - {end_t}s]")
    print("Baseline PoC Dataset vs New Optimized Dataset")
    print("=========================================================================\n")
    
    # We slice both strictly to ensure absolute parity (Baseline should already be sliced, but we do it anyway for mathematical safety)
    # We also enforce that the Optimized Simulation ONLY evaluates the exact same vehicles present in the Baseline PoC.
    baseline_metrics, valid_basenames = get_sliced_dataset_metrics(baseline_dir, "Baseline PoC", start_t, end_t)
    optimized_metrics, _ = get_sliced_dataset_metrics(args.optimized_dir, "Optimized Simulation", start_t, end_t, target_basenames=valid_basenames)
    
    # Generate Markdown Table
    md_lines = []
    md_lines.append("# AI-Assisted Trajectory Optimization Results")
    md_lines.append("")
    md_lines.append("This report compares the baseline network performance during peak congestion against the new simulation powered by Transformer reachability decisions.")
    md_lines.append("")
    md_lines.append("## Dataset Constraints (Anti-Leakage)")
    md_lines.append("To ensure a fair mathematical comparison and prevent data leakage, the new optimized simulation dataset was strictly filtered in memory to match the exact temporal bounds AND the exact specific vehicles extracted in the original Proof-of-Concept peak congestion window.")
    md_lines.append(f"- **Window Start:** {start_t} s")
    md_lines.append(f"- **Window End:** {end_t} s")
    md_lines.append(f"- **Duration:** {config.get('duration_sec')} s")
    md_lines.append(f"- **Specific Vehicles Evaluated:** {len(valid_basenames)}")
    md_lines.append("")
    md_lines.append("## Metrics Comparison")
    md_lines.append("| Metric | Baseline PoC (Unoptimized) | Optimized Simulation (AI Decisions) | Improvement |")
    md_lines.append("|--------|----------------------------|-------------------------------------|-------------|")
    
    for key in baseline_metrics.keys():
        val_base = baseline_metrics[key]
        val_opt = optimized_metrics[key]
        
        # Calculate improvement for delay and PLR
        improvement = "-"
        if "Delay" in key or "Loss" in key:
            try:
                num_base = float(val_base.split()[0].replace(',', ''))
                num_opt = float(val_opt.split()[0].replace(',', ''))
                if num_base > 0:
                    pct_change = ((num_base - num_opt) / num_base) * 100
                    if pct_change > 0:
                        improvement = f"📉 -{pct_change:.1f}%"
                    elif pct_change < 0:
                        improvement = f"📈 +{abs(pct_change):.1f}%"
                    else:
                        improvement = "0.0%"
            except:
                pass
                
        md_lines.append(f"| {key} | {val_base} | {val_opt} | {improvement} |")
        
    report_path = os.path.join(os.path.dirname(args.config), "poc_vs_optimized_comparison.md")
    
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"\nComparison complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()
