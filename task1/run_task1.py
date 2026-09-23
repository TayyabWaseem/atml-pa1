import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run_script(script_path):
    print(f"\n{'='*80}\nRunning {script_path}\n{'='*80}")
    # Run using the same python executable
    cmd = [sys.executable, str(REPO_ROOT / script_path)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(result.returncode)

def main():
    scripts = [
        "task1/data/make_subset.py",
        "task1/analysis/train_heads.py",
        "task1/data/make_cue_conflicts.py",
        "task1/analysis/evaluate_bias.py",
        "task1/analysis/feature_similarity.py",
        "task1/analysis/representation.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\nTask 1 execution complete. All results saved in task1/results/")

if __name__ == "__main__":
    main()

