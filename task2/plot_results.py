import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_main_results(results, out_dir):
    methods = [m for m in ["Source-only", "DAN", "DANN", "CDAN"] if m in results]
    if not methods: return
    
    source_accs = [results[m]["mean_source_acc"] for m in methods]
    target_accs = [results[m]["target_acc"] for m in methods]
    domain_seps = [results[m]["domain_separability"] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, source_accs, width, label='Mean Source Acc', color='skyblue')
    ax.bar(x, target_accs, width, label='Target Acc (Sketch)', color='salmon')
    ax.bar(x + width, domain_seps, width, label='Domain Separability', color='lightgreen')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Task 2: UDA Main Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'main_comparison.png', dpi=300)
    plt.close()

def plot_controlled_study(results, out_dir):
    grls = [0.25, 0.5, 1.0]
    keys = ["DANN_grl0.25", "DANN_grl0.5", "DANN"]
    
    valid_keys = [k for k in keys if k in results]
    if len(valid_keys) < 3: return
    
    source_accs = [results[k]["mean_source_acc"] for k in valid_keys]
    target_accs = [results[k]["target_acc"] for k in valid_keys]
    domain_seps = [results[k]["domain_separability"] for k in valid_keys]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Max Gradient Reversal Strength (GRL)')
    ax1.set_ylabel('Classification Accuracy (%)', color=color)
    ax1.plot(grls, source_accs, marker='o', label='Mean Source Acc', color=color, linestyle='--')
    ax1.plot(grls, target_accs, marker='s', label='Target Acc (Sketch)', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='center left')
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Domain Separability (%)', color=color)
    ax2.plot(grls, domain_seps, marker='^', label='Domain Separability', color=color, linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='center right')
    
    plt.title('Controlled Design Study: DANN GRL Strength')
    plt.tight_layout()
    plt.savefig(out_dir / 'controlled_study_dann.png', dpi=300)
    plt.close()

def main():
    res_path = Path("task2/results/evaluation.json")
    if not res_path.exists():
        print("evaluation.json not found!")
        return
        
    with open(res_path, "r") as f:
        results = json.load(f)
        
    out_dir = Path("task2/results/figures")
    out_dir.mkdir(exist_ok=True)
    
    plot_main_results(results, out_dir)
    plot_controlled_study(results, out_dir)
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
