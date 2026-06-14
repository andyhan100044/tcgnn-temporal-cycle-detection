"""
Robust Elliptic1 downloader for Kaggle.

Usage:
    # 1. Get your Kaggle API token from https://www.kaggle.com/settings/account
    #    Click "Create New Token" → downloads kaggle.json
    # 2. Place it at C:\\Users\\<you>\\.kaggle\\kaggle.json  (Windows)
    #                   or ~/.kaggle/kaggle.json           (Linux/Mac)
    # 3. Run: python data/download_elliptic1.py

This script:
  - Validates kaggle.json is present and has 600 perms
  - Downloads elliptic-data-set via Kaggle CLI
  - Unzips and copies the 3 CSV files to data/elliptic1/raw/
  - Verifies file sizes (features ~99MB, edgelist ~31MB, classes ~6MB)
  - Auto-runs preprocess_elliptic1.py to produce data/elliptic1/processed/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_FILES = {
    "elliptic_txs_features.csv": 95_000_000,   # ~99 MB
    "elliptic_txs_edgelist.csv":  30_000_000,   # ~31 MB
    "elliptic_txs_classes.csv":    6_000_000,   # ~6 MB
}


def find_kaggle_json() -> Path | None:
    """Look for kaggle.json in standard locations."""
    candidates = [
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
        Path("C:/Users/Administrator/.kaggle/kaggle.json"),
        Path("E:/kaggle.json"),
        Path("E:/000001research/kaggle.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def validate_token(path: Path) -> tuple[str, str] | None:
    """Parse and return (username, key) from kaggle.json."""
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("username"), data.get("key")
    except Exception as e:
        print(f"[ERR] Failed to parse {path}: {e}")
        return None


def install_kaggle_config(path: Path) -> None:
    """Copy kaggle.json to ~/.kaggle/kaggle.json with 600 perms."""
    target_dir = Path.home() / ".kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "kaggle.json"
    if not target.exists() or target.read_text() != path.read_text():
        shutil.copy(path, target)
        print(f"[OK] Copied {path} -> {target}")
    if sys.platform != "win32":
        os.chmod(target, 0o600)


def download_via_cli(target_dir: Path) -> bool:
    """Run `kaggle datasets download` and return success."""
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "ellipticco/elliptic-data-set",
        "-p", str(target_dir),
        "--unzip",
    ]
    print(f"[CMD] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[ERR] Kaggle CLI failed:\n{result.stderr}")
            return False
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("[ERR] 'kaggle' CLI not found. Run: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("[ERR] Kaggle download timed out (>10 min)")
        return False


def verify_files(raw_dir: Path) -> bool:
    """Check that all expected files exist with reasonable sizes."""
    ok = True
    for fname, expected_min in EXPECTED_FILES.items():
        f = raw_dir / fname
        if not f.exists():
            print(f"[ERR] Missing: {f}")
            ok = False
            continue
        size = f.stat().st_size
        if size < expected_min:
            print(f"[ERR] {fname}: {size:,} bytes (expected >{expected_min:,})")
            ok = False
        else:
            print(f"[OK] {fname}: {size:,} bytes")
    return ok


def main():
    raw_dir = Path("data/elliptic1/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    token_path = find_kaggle_json()
    if token_path is None:
        print("[FAIL] No kaggle.json found.")
        print()
        print("To proceed, ONE of the following:")
        print("  (a) Provide your Kaggle API token:")
        print("      1. Visit https://www.kaggle.com/settings/account")
        print("      2. Click 'Create New Token' → saves kaggle.json")
        print("      3. Place kaggle.json at any of:")
        for c in [
            Path.home() / ".kaggle" / "kaggle.json",
            Path("C:/Users/Administrator/.kaggle/kaggle.json"),
            Path("E:/000001research/kaggle.json"),
        ]:
            print(f"         - {c}")
        print()
        print("  (b) Manually download Elliptic1:")
        print("      1. Visit https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("      2. Download the 3 CSVs")
        print(f"      3. Place them in {raw_dir.absolute()}/")
        print()
        print("Expected files:")
        for f, sz in EXPECTED_FILES.items():
            print(f"  - {f}  (~{sz//1_000_000} MB)")
        sys.exit(1)

    creds = validate_token(token_path)
    if creds is None:
        sys.exit(1)
    username, key = creds
    print(f"[OK] Token found: {token_path} (user={username})")
    install_kaggle_config(token_path)

    # Download
    if not download_via_cli(raw_dir):
        print("[FAIL] Download failed. Check network / Kaggle status.")
        sys.exit(1)

    # Verify
    if not verify_files(raw_dir):
        print("[FAIL] Verification failed.")
        sys.exit(1)

    print()
    print("[DONE] Elliptic1 raw files in place. Run:")
    print("       python data/preprocess_elliptic1.py")
    print("       python experiments/run_experiment.py --full")


if __name__ == "__main__":
    main()