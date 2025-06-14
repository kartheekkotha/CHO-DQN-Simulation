import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

map_y_labels = {
    'total_Ho_count':'HOF Count',
    'rlf':'RLF Count'
    }
map_x_labels = {
    'N310_THRESHOLD':'N310 Threshold',
    'Oexec' : 'O_exec (dB)',
    'Texec' : 'T_exec (ms)',
    'Oprep' : 'O_prep (dB)',
    'Tprep' : 'T_prep (ms)',
    'T310' : 'T310 (ms)',
}
def plot_comparison_bars(parent_folder, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(parent_folder, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each parameter folder
    for param_folder in os.listdir(parent_folder):
        param_path = os.path.join(parent_folder, param_folder)
        if not os.path.isdir(param_path) or param_folder == os.path.basename(output_dir):
            continue

        print(f"Processing parameter: {param_folder}")
        data = defaultdict(lambda: { 'before_training': None, 'after_training': None })

        # Read CSVs
        for fname in os.listdir(param_path):
            if not fname.endswith('.csv'):
                continue
            # Expect format: <param>_<metric>_<config>.csv
            # config is before_training or after_training
            base = fname[:-4]
            if base.endswith('_before_training'):
                config = 'before_training'
                core = base[:-len('_before_training')]
            elif base.endswith('_after_training'):
                config = 'after_training'
                core = base[:-len('_after_training')]
            else:
                print(f"Skipping unknown config file: {fname}")
                continue
            parts = core.split('_', 1)
            if len(parts) != 2:
                print(f"Malformed filename (metric parse): {fname}")
                continue
            _, metric = parts

            # Load CSV
            path_csv = os.path.join(param_path, fname)
            with open(path_csv, newline='') as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)
                x_vals = [float(r[0]) for r in rows]
                y_vals = [float(r[1]) for r in rows]

            data[metric][config] = (x_vals, y_vals)

        # Prepare subfolder for plots
        out_sub = os.path.join(output_dir, param_folder)
        os.makedirs(out_sub, exist_ok=True)
        print(data)
        # Plot each metric
        for metric, cfgs in data.items():
            matched_label = None
            for key in map_y_labels:
                if key in metric:
                    matched_label = map_y_labels[key]
                    break
            y_label = matched_label if matched_label else metric.replace('_', ' ').title()
            if param_folder in map_x_labels:
                x_label = map_x_labels[param_folder]
            else:
                x_label = param_folder
            b = cfgs['before_training']
            a = cfgs['after_training']
            if b is None or a is None:
                print(f"Skipping {metric}: missing before or after data")
                continue
            x_b, y_b = b
            x_a, y_a = a
            if x_b != x_a:
                print(f"Skipping {metric}: x-values mismatch")
                continue

            # Bar chart
            ind = range(len(x_b))
            w = 0.35
            plt.figure(figsize=(10,6))
            plt.bar([i - w/2 for i in ind], y_b, w, label='Conventional CHO')
            plt.bar([i + w/2 for i in ind], y_a, w, label='CHO+ Proposed power control')
            plt.xticks(ind, [str(int(x)) if x.is_integer() else str(x) for x in x_b])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{y_label} Comparison")
            plt.legend(fontsize=12)
            plt.grid(True)

            out_png = os.path.join(out_sub, f"{x_label}_{metric}_comparison.png")
            print(f"Saving plot: {out_png}")
            plt.savefig(out_png)
            plt.close()

if __name__ == '__main__':
    parent_folder = 'test_logs_overall/2025-05-18_22-39-01'  # adjust
    output_dir ='final_graphs'
    plot_comparison_bars(parent_folder, output_dir)
