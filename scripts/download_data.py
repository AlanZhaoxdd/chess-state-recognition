"""Download the rendered chessboard dataset into ``data/render/``.

The dataset is the synthetic one published with Wölflein & Arandjelović,
"Determining Chess Game State from an Image" (J. Imaging, 2021), hosted on
the Open Science Framework: https://doi.org/10.17605/OSF.IO/XF3KA

Usage::

    python scripts/download_data.py                 # val + test (about 0.75 GB)
    python scripts/download_data.py --splits val    # one split only
    python scripts/download_data.py --splits train val test   # everything (about 7.5 GB)

Downloads are resumable-by-rerun: a finished archive with the right checksum
is not downloaded again, and a split that is already unpacked is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from chessrec import config

#: Archive name -> (download URL, size in bytes, MD5), as listed by OSF.
ARCHIVES = {
    "val.zip": ("https://osf.io/download/ydtcs/", 222_952_872,
                "79fcba30484864b3ef745eaf1676ec10"),
    "test.zip": ("https://osf.io/download/7p69s/", 522_129_163,
                 "167938fc264bd60f99f15c3054fbbc55"),
    "train.zip": ("https://osf.io/download/z5vgm/", 2_416_209_370,
                  "e992d3982abfd0103614fc61725d10df"),
    "train.z01": ("https://osf.io/download/np56u/", 4_294_967_296,
                  "037e9de49a0b10f453d3d0da8cf28609"),
}

#: The training set is one ZIP split into two parts.
SPLIT_ARCHIVES = {
    "val": ["val.zip"],
    "test": ["test.zip"],
    "train": ["train.z01", "train.zip"],
}

CHUNK = 1 << 20  # 1 MiB


def md5(path: Path) -> str:
    """MD5 checksum of a file, read in chunks so large files fit in memory."""
    digest = hashlib.md5()
    with open(path, "rb") as f, tqdm(
            total=path.stat().st_size, unit="B", unit_scale=True,
            desc=f"verify   {path.name}", leave=False) as bar:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def download(name: str, folder: Path) -> Path:
    """Download one archive (unless a verified copy is already there)."""
    url, size, checksum = ARCHIVES[name]
    target = folder / name
    if target.exists() and target.stat().st_size == size and md5(target) == checksum:
        print(f"{name}: already downloaded")
        return target

    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as f, tqdm(
            total=size, unit="B", unit_scale=True, desc=f"download {name}") as bar:
        while chunk := response.read(CHUNK):
            f.write(chunk)
            bar.update(len(chunk))

    if md5(partial) != checksum:
        raise RuntimeError(f"{name}: checksum mismatch, delete {partial} and retry")
    partial.replace(target)
    return target


def unzip(archive: Path, folder: Path) -> None:
    """Extract a normal (single-file) ZIP archive into ``folder``."""
    with zipfile.ZipFile(archive) as z:
        for member in tqdm(z.infolist(), desc=f"extract  {archive.name}", unit="file"):
            z.extract(member, folder)


def unzip_split_train(folder: Path) -> bool:
    """Extract the two-part training archive with an external tool.

    Python's ``zipfile`` cannot read split archives, so this needs 7-Zip
    (``7z``), Bandizip (``bz``) or Info-ZIP (``zip`` + ``unzip``).

    Returns:
        True if the archive was extracted.
    """
    archive = folder / "train.zip"
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if seven_zip:
        # 7-Zip finds train.z01 next to train.zip by itself.
        return subprocess.run([seven_zip, "x", "-y", f"-o{folder}", str(archive)]).returncode == 0

    # Bandizip's console tool. It is usually not on the PATH: add its install
    # folder to the PATH, or set the BANDIZIP environment variable to bz.exe.
    bandizip = (shutil.which("bz") or os.environ.get("BANDIZIP")
                or r"C:\Program Files\Bandizip\bz.exe")
    if Path(bandizip).is_file():
        return subprocess.run([bandizip, "x", "-y", f"-o:{folder}", str(archive)]).returncode == 0

    if shutil.which("zip") and shutil.which("unzip"):
        merged = folder / "train_full.zip"
        if subprocess.run(["zip", "-s", "0", str(archive), "--out", str(merged)]).returncode == 0:
            ok = subprocess.run(["unzip", "-q", "-o", str(merged), "-d", str(folder)]).returncode == 0
            merged.unlink(missing_ok=True)
            return ok
    return False


def has_data(split_folder: Path) -> bool:
    """True if a split folder already holds label files."""
    return split_folder.is_dir() and any(split_folder.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", choices=config.SPLITS,
                        default=["val", "test"],
                        help="splits to download (default: val test)")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .zip files after extracting")
    args = parser.parse_args()

    folder = config.RENDER_DIR
    folder.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if has_data(folder / split):
            print(f"{split}: already extracted in {folder / split}")
            continue

        archives = [download(name, folder) for name in SPLIT_ARCHIVES[split]]
        if split == "train":
            if not unzip_split_train(folder):
                print(
                    "\nCould not extract the training set automatically.\n"
                    f"train.zip and train.z01 in {folder} are two parts of one archive.\n"
                    "Install 7-Zip (https://www.7-zip.org), open train.zip with it and\n"
                    f"extract to {folder} so that {folder / 'train'} is created.",
                    file=sys.stderr)
                return 1
        else:
            unzip(archives[0], folder)

        if not has_data(folder / split):
            print(f"{split}: extracted, but no labels found in {folder / split}; "
                  f"check the folder layout.", file=sys.stderr)
            return 1
        if not args.keep_archives:
            for archive in archives:
                archive.unlink()
        print(f"{split}: ready in {folder / split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
