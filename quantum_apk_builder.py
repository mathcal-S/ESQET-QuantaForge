#!/usr/bin/env python3
"""
Gradle plugin hook for quantum-secured APK builds
"""

from esqet_master import PostQuantumAPKOracle
import subprocess
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python quantum_apk_builder.py <apk_path>")
        sys.exit(1)
    
    apk_path = sys.argv[1]
    oracle = PostQuantumAPKOracle()
    
    validation = oracle.validate_apk_signature_quantum(apk_path)
    
    if validation["approved"]:
        # Sign with Dilithium
        signed_apk = oracle.sign_apk_dilithium(apk_path)
        print(f"✅ Quantum-signed APK: {signed_apk}")
        sys.exit(0)
    else:
        print(f"❌ Build rejected: F_QC={validation['F_QC']:.3f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
