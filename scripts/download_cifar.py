"""
scripts/download_cifar.py
--------------------------
Resume-capable CIFAR-10 / CIFAR-100 downloader.
Falls back to torchvision if the file is already complete.
"""
import os
import sys
import tarfile
import time
from urllib.request import Request, urlopen

DATASETS = {
    "cifar10": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "filename": "cifar-10-python.tar.gz",
        "expected_size": 170498071,
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "filename": "cifar-100-python.tar.gz",
        "expected_size": 169001437,
    },
}

def download(name="cifar10", dest_dir="data/raw"):
    info = DATASETS[name]
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, info["filename"])
    expected = info["expected_size"]

    existing = os.path.getsize(dest) if os.path.exists(dest) else 0

    if existing >= expected:
        print(f"[{name}] Already downloaded ({existing/1e6:.1f} MB). Skipping.")
    else:
        req = Request(info["url"])
        if existing:
            req.add_header("Range", f"bytes={existing}-")
            print(f"[{name}] Resuming from {existing/1e6:.1f} MB ...")
        else:
            print(f"[{name}] Starting download ...")

        with urlopen(req, timeout=60) as resp:
            mode = "ab" if existing else "wb"
            downloaded = existing
            chunk_size = 131072
            t0 = time.time()
            with open(dest, mode) as f:
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    elapsed = max(time.time() - t0, 1e-6)
                    speed = (downloaded - existing) / elapsed / 1024
                    pct = downloaded / expected * 100
                    sys.stdout.write(
                        f"\r  {pct:5.1f}%  {downloaded/1e6:6.1f}/{expected/1e6:.0f} MB  "
                        f"{speed:6.0f} kB/s   "
                    )
                    sys.stdout.flush()
        print(f"\n[{name}] Download complete: {downloaded/1e6:.1f} MB")

    # Extract
    marker_dir = os.path.join(dest_dir, "cifar-10-batches-py" if name == "cifar10" else "cifar-100-python")
    if os.path.exists(marker_dir):
        print(f"[{name}] Already extracted.")
    else:
        print(f"[{name}] Extracting ...")
        with tarfile.open(dest, "r:gz") as tf:
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                tf.extractall(dest_dir)
        print(f"[{name}] Extraction complete.")

    print(f"[{name}] Ready in {dest_dir}/")

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    download(name)