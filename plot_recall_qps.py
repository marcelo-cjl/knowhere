#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Build times provided by user (in seconds)
BUILD_TIMES = {
    'HNSWLIB': 1020.92,
    'HNSW_SQ': 462.36,
    'GPU_VAMANA_iter1': 116.09,
    'GPU_VAMANA_iter2': 293.54,
    'GPU_CAGRA_iter20': 24.12,
    'GPU_CAGRA_iter40': 38.07,
    'GPU_CAGRA_iter60': 53.82,
    'GPU_CAGRA_iter80': 67.32,
    'GPU_CAGRA_iter100': 82.45,
}

# BUILD_TIMES = {
#     'HNSWLIB': 1112.7335,
#     'HNSW_SQ': 348.2626,
#     'GPU_VAMANA_iter1': 123.6187,
#     'GPU_VAMANA_iter2': 301.7548,
#     'GPU_CAGRA_iter20': 23.14,
#     'GPU_CAGRA_iter40': 36.13,
#     'GPU_CAGRA_iter60': 49.81,
#     'GPU_CAGRA_iter80': 62.02,
#     'GPU_CAGRA_iter100': 76.75,
# }

def parse_result_file(filepath):
    """Parse result.txt and return data grouped by index type."""
    data = defaultdict(list)

    # Track iter for indices without explicit iter in name
    cagra_iter_list = [20, 40, 60, 80, 100]
    vamana_iter_list = [1, 2]
    cagra_count = 0
    vamana_count = 0
    last_cagra_ef = None
    last_vamana_ef = None

    with open(filepath, 'r') as f:
        for line in f:
            # Skip header and comment lines
            if (line.startswith('=') or line.startswith('#') or
                line.startswith('Dataset') or line.startswith('---') or
                line.startswith('Date:') or not line.strip()):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            # Try to parse as data line
            try:
                dataset = parts[0]
                index_name = parts[1]
                # Skip if first part doesn't look like a dataset name
                if dataset not in ['gist', 'cohere', 'sift', 'glove']:
                    continue

                qps = float(parts[3])
                recall = float(parts[4])
            except (ValueError, IndexError):
                continue

            # Group by index type
            if 'GPU_CAGRA' in index_name:
                # Check if iter is in name
                m = re.match(r'GPU_CAGRA_iter(\d+)', index_name)
                if m:
                    group = f"GPU_CAGRA_iter{m.group(1)}"
                else:
                    # No iter in name, determine from order
                    ef_match = re.search(r'ef(\d+)', index_name)
                    if ef_match:
                        current_ef = int(ef_match.group(1))
                        # If ef decreased or is first entry, we moved to next iter
                        if last_cagra_ef is not None and current_ef <= last_cagra_ef:
                            cagra_count += 1
                        last_cagra_ef = current_ef
                        iter_idx = cagra_count % len(cagra_iter_list)
                        group = f"GPU_CAGRA_iter{cagra_iter_list[iter_idx]}"
                    else:
                        group = 'GPU_CAGRA'

            elif 'GPU_VAMANA' in index_name:
                m = re.match(r'GPU_VAMANA_iter(\d+)', index_name)
                if m:
                    group = f"GPU_VAMANA_iter{m.group(1)}"
                else:
                    ef_match = re.search(r'ef(\d+)', index_name)
                    if ef_match:
                        current_ef = int(ef_match.group(1))
                        if last_vamana_ef is not None and current_ef <= last_vamana_ef:
                            vamana_count += 1
                        last_vamana_ef = current_ef
                        iter_idx = vamana_count % len(vamana_iter_list)
                        group = f"GPU_VAMANA_iter{vamana_iter_list[iter_idx]}"
                    else:
                        group = 'GPU_VAMANA'

            elif 'HNSW_SQ' in index_name:
                group = 'HNSW_SQ'
            elif 'HNSWLIB' in index_name:
                group = 'HNSWLIB'
            else:
                group = index_name

            data[group].append((recall, qps))

    return data

def get_label_with_build_time(group):
    """Get label with build time for legend."""
    build_time = BUILD_TIMES.get(group, 0)
    if build_time == 0:
        return group
    elif build_time >= 60:
        return f"{group} (build: {build_time:.1f}s)"
    else:
        return f"{group} (build: {build_time:.2f}s)"

def plot_recall_qps(data, output_file='recall_qps_plot.png'):
    """Plot recall vs QPS curves - two separate plots."""

    # Define colors and markers
    colors = {
        'GPU_CAGRA_iter20': '#1f77b4',
        'GPU_CAGRA_iter40': '#2ca02c',
        'GPU_CAGRA_iter60': '#ff7f0e',
        'GPU_CAGRA_iter80': '#d62728',
        'GPU_CAGRA_iter100': '#9467bd',
        'GPU_VAMANA_iter1': '#8c564b',
        'GPU_VAMANA_iter2': '#e377c2',
        'HNSW_SQ': '#17becf',
        'HNSWLIB': '#7f7f7f',
    }

    markers = {
        'GPU_CAGRA_iter20': 'o',
        'GPU_CAGRA_iter40': 's',
        'GPU_CAGRA_iter60': '^',
        'GPU_CAGRA_iter80': 'D',
        'GPU_CAGRA_iter100': 'v',
        'GPU_VAMANA_iter1': 'p',
        'GPU_VAMANA_iter2': 'h',
        'HNSW_SQ': '*',
        'HNSWLIB': 'X',
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ===== Plot 1: HNSW_SQ only =====
    if 'HNSW_SQ' in data:
        points = data['HNSW_SQ']
        points.sort(key=lambda x: x[0])
        recalls = [p[0] for p in points]
        qps_values = [p[1] for p in points]

        label = get_label_with_build_time('HNSW_SQ')
        ax1.plot(recalls, qps_values,
                 marker='*',
                 color='#17becf',
                 label=label,
                 linewidth=2,
                 markersize=10)

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('QPS', fontsize=12)
    ax1.set_title('HNSW_SQ (SQ4U + FP16 Refine)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.76, 1.0)

    # Auto-scale Y axis based on data
    if 'HNSW_SQ' in data:
        max_qps = max(p[1] for p in data['HNSW_SQ'])
        ax1.set_ylim(0, max_qps * 1.1)

    # ===== Plot 2: All other indices =====
    other_groups = ['GPU_CAGRA_iter20', 'GPU_CAGRA_iter40', 'GPU_CAGRA_iter60',
                    'GPU_CAGRA_iter80', 'GPU_CAGRA_iter100',
                    'GPU_VAMANA_iter1', 'GPU_VAMANA_iter2', 'HNSWLIB']

    max_qps_other = 0
    for group in other_groups:
        if group not in data:
            continue
        points = data[group]
        points.sort(key=lambda x: x[0])
        recalls = [p[0] for p in points]
        qps_values = [p[1] for p in points]

        max_qps_other = max(max_qps_other, max(qps_values))

        color = colors.get(group, '#000000')
        marker = markers.get(group, 'o')
        label = get_label_with_build_time(group)

        ax2.plot(recalls, qps_values,
                 marker=marker,
                 color=color,
                 label=label,
                 linewidth=2,
                 markersize=6)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('QPS', fontsize=12)
    ax2.set_title('GPU_CAGRA / GPU_VAMANA / HNSWLIB', fontsize=14)
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.82, 1.0)
    ax2.set_ylim(0, max_qps_other * 1.1 if max_qps_other > 0 else 2500)

    plt.suptitle('Recall vs QPS - Benchmark Results(Gist)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print("\n=== Summary at ~95% recall ===")
    all_groups = ['HNSW_SQ'] + other_groups
    for group in all_groups:
        if group not in data:
            continue
        points = data[group]
        closest = min(points, key=lambda x: abs(x[0] - 0.95))
        build_time = BUILD_TIMES.get(group, 0)
        if build_time > 0:
            print(f"{group:25s}: recall={closest[0]:.4f}, qps={closest[1]:7.0f}, build={build_time:.2f}s")
        else:
            print(f"{group:25s}: recall={closest[0]:.4f}, qps={closest[1]:7.0f}")

if __name__ == '__main__':
    data = parse_result_file('/home/ubuntu/knowhere/result.txt')

    # Print detected groups
    print("Detected index groups:")
    for group in sorted(data.keys()):
        print(f"  {group}: {len(data[group])} data points")
    print()

    plot_recall_qps(data, '/home/ubuntu/knowhere/recall_qps_plot.png')
