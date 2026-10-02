"""
Download required model weights.
Run this once before using the pipeline.
"""

import os
import sys
import hashlib
from pathlib import Path
import urllib.request

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = {
    "depth_anything_v2_vits.pth": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "size_mb": 99,
        "description": "Depth Anything V2 (ViT-Small) — for video tier monocular depth",
    },
}


def download_with_progress(url: str, dest: Path, description: str = ""):
    """Download a file with a progress bar."""
    print(f"\nDownloading {description}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, downloaded / total_size * 100)
            bar = "#" * int(pct / 2)
            print(f"\r  [{bar:<50}] {pct:.0f}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
    print(f"\n  Done: {dest}")


def main():
    print("=" * 60)
    print("  Brynz Model Download Script")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if HuggingFace CLI is available
    hf_available = False
    try:
        import huggingface_hub
        hf_available = True
    except ImportError:
        pass
    
    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        
        if dest.exists():
            print(f"\n  ✅ {filename} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
            continue
        
        print(f"\n  Downloading {filename} (~{info['size_mb']} MB)")
        print(f"  {info['description']}")
        
        try:
            if hf_available and "huggingface.co" in info["url"]:
                from huggingface_hub import hf_hub_download
                # Parse repo and filename from URL
                parts = info["url"].replace("https://huggingface.co/", "").split("/")
                repo_id = "/".join(parts[:2])
                hf_filename = parts[-1]
                
                hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    local_dir=str(MODELS_DIR),
                )
            else:
                download_with_progress(info["url"], dest, info["description"])
        except Exception as e:
            print(f"\n  ERROR downloading {filename}: {e}")
            print(f"  Please download manually from: {info['url']}")
    
    print("\n" + "=" * 60)
    print("  Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
