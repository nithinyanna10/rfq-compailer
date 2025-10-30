import argparse
from pathlib import Path
import urllib.request

def fetch(urls_file: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for line in Path(urls_file).read_text().splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        name = url.split("/")[-1] or "download.pdf"
        dest = Path(out_dir) / name
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded {url} -> {dest}")
        except Exception as e:
            print(f"Failed {url}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    fetch(args.urls_file, args.out_dir)
