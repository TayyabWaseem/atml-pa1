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
    print("Starting Task 3 Smoke Test...")
    
    methods = [
        ["--method", "dan_dg", "--lambda_dg", "1.0"],
        ["--method", "sam"],
        ["--method", "dan_dg", "--lambda_dg", "0.1"],
        ["--method", "dan_dg", "--lambda_dg", "10.0"],
    ]
    
    for method_args in methods:
        cmd = [sys.executable, "task3/train.py"] + method_args + ["--max_epochs", "1", "--limit_batches", "2"]
        run_cmd(cmd)
        
    print("\nTraining smoke test complete! Now testing evaluation...")
    run_cmd([sys.executable, "task3/evaluate_sketch.py"])
    
    print("\nSUCCESS! Everything ran without errors.")

if __name__ == "__main__":
    main()
