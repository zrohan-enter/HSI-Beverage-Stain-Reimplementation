from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "salinas"
BASE_URL = "https://zenodo.org/records/15771735/files"
FILES = {
    "Salinas_corrected.mat": "485d8802f4a6b4ebc0767d48dd0da06b",
    "Salinas_gt.mat": "7b8da653a61bb0271b27b37fb926390f",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, expected_md5 in FILES.items():
        destination = OUTPUT_DIR / filename
        temporary = destination.with_suffix(destination.suffix + ".part")

        if destination.exists() and md5(destination) == expected_md5:
            print(f"Already verified: {destination}")
            continue

        url = f"{BASE_URL}/{filename}?download=1"
        print(f"Downloading {url}")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 HSI-Reimplementation/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as file:
                shutil.copyfileobj(response, file)

        actual = md5(temporary)
        if actual != expected_md5:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}: {actual} != {expected_md5}"
            )

        temporary.replace(destination)
        print(f"Verified: {destination}")

    print("Salinas download and verification completed.")


if __name__ == "__main__":
    main()
