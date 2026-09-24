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
    print("Starting Task 4 Smoke Test...")
    
    methods = [
        ["--method", "vanilla"],
        ["--method", "gcsc"],
        ["--method", "proser"],
    ]
    
    # Train
    for method_args in methods:
        cmd = [sys.executable, "task4/train.py"] + method_args + ["--max_epochs", "1", "--limit_batches", "2"]
        run_cmd(cmd)
        
    # Extract
    for method_args in methods:
        cmd = [sys.executable, "task4/extract_outputs.py"] + method_args
        run_cmd(cmd)
        
    # Evaluate & Plot
    print("\nTraining & Extraction complete! Now testing evaluation...")
    run_cmd([sys.executable, "task4/evaluate_osr.py"])
    run_cmd([sys.executable, "task4/plot_osr.py"])
    
    print("\nSUCCESS! Everything ran without errors.")

if __name__ == "__main__":
    main()
