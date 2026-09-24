import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def run_cmd(cmd):
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("Starting Task 2 Smoke Test...")
    print("This will run all models for just 1 epoch and 2 batches per epoch to ensure the pipeline is bug-free.")
    
    methods = [
        ["--method", "source_only"],
        ["--method", "dan"],
        ["--method", "dann", "--max_grl", "1.0"],
        ["--method", "cdan"],
        ["--method", "dann", "--max_grl", "0.25"],
        ["--method", "dann", "--max_grl", "0.5"],
    ]
    
    for method_args in methods:
        cmd = [sys.executable, "task2/train.py"] + method_args + ["--max_epochs", "1", "--limit_batches", "2"]
        run_cmd(cmd)
        
    print("\nTraining smoke test complete! Now testing evaluation...")
    run_cmd([sys.executable, "task2/evaluate_final.py"])
    
    print("\nEvaluation complete! Now testing plotting...")
    run_cmd([sys.executable, "task2/plot_results.py"])
    
    print("\nSUCCESS! Everything ran without errors.")

if __name__ == "__main__":
    main()
