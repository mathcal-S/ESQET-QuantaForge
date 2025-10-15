#!/usr/bin/env python3
"""
Package ESQET artifacts for Termux deployment
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
import argparse

def package_esqet_artifacts(quantum: bool = True, platform: str = "android-arm64"):
    """Build and package ESQET artifacts"""
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Pre-compute quantum circuits
    if quantum:
        print("🔮 Pre-computing quantum circuits...")
        subprocess.run(["python", "src/precompute_circuits.py", "--output", "quantum_circuits.json"])
    
    # Copy runtime files
    runtime_files = [
        "src/esqet_core.py",
        "requirements-runtime.txt",
        "quantum_circuits.json" if quantum else []
    ]
    for f in runtime_files:
        if os.path.exists(f):
            shutil.copy(f, "dist/")
    
    # Package wheels (mock for now; Actions will build real)
    wheels_dir = dist_dir / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    
    # Generate deployment guide
    with open("dist/DEPLOYMENT.md", "w") as f:
        f.write("""
# ESQET Deployment Guide

## Termux Zero-Dep Install
1. Extract artifacts
2. `pip install --find-links ./wheels/ -r requirements-runtime.txt`
3. `python esqet_core.py --generate-nft "Test"`

## Offline Quantum
Uses cached circuits—no Qiskit needed.

## Validation
F_QC > 1.5 = Coherent
        """)
    
    # Create ZIP
    shutil.make_archive("dist/esqet-quantum-artifacts", 'zip', dist_dir)
    print("✅ Artifacts packaged: dist/esqet-quantum-artifacts.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum", action="store_true")
    parser.add_argument("--platform", default="android-arm64")
    args = parser.parse_args()
    package_esqet_artifacts(args.quantum, args.platform)
