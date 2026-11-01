"""Download the Coswara batches listed in config.BATCHES and extract the heavy-cough recordings.

Coswara stores each batch as split tar.gz parts in a Git repo. A sparse, blob-less clone fetches
only the batches we need (~0.6 GB) instead of the whole repository.

    python -m src.data.download_coswara
"""
import subprocess, tarfile, io
from src import config as C

REPO_URL = "https://github.com/iiscleap/Coswara-Data.git"


def sh(*args, cwd=None):
    subprocess.run(list(args), cwd=cwd, check=True)


def clone():
    if (C.COSWARA_REPO / ".git").exists():
        return
    C.DATA.mkdir(parents=True, exist_ok=True)
    sh("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(C.COSWARA_REPO))


def extract(batch):
    out = C.RAW / batch
    if out.exists() and any(out.iterdir()):
        print(f"{batch}: already extracted")
        return
    sh("git", "sparse-checkout", "add", batch, cwd=C.COSWARA_REPO)
    parts = sorted((C.COSWARA_REPO / batch).glob(f"{batch}.tar.gz.*"))
    blob = b"".join(p.read_bytes() for p in parts)              # parts are a byte-split archive
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("/cough-heavy.wav")]
        tar.extractall(C.RAW, members=members, filter="data")
    print(f"{batch}: {len(list(out.iterdir()))} participants")
    for p in parts:                                              # free disk space
        p.unlink()


if __name__ == "__main__":
    clone()
    for b in C.BATCHES:
        extract(b)
    print("metadata:", C.COSWARA_REPO / "combined_data.csv")
