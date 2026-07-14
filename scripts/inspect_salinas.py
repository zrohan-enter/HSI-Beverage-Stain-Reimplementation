from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "salinas"
CUBE_PATH = DATASET_DIRECTORY / "Salinas_corrected.mat"
LABEL_PATH = DATASET_DIRECTORY / "Salinas_gt.mat"

EXPECTED_CUBE_MD5 = "485d8802f4a6b4ebc0767d48dd0da06b"
EXPECTED_LABEL_MD5 = "7b8da653a61bb0271b27b37fb926390f"


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_expected_array(
    path: Path,
    preferred_key: str,
    expected_dimensions: int,
) -> tuple[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    contents = loadmat(path)

    if preferred_key in contents:
        array = contents[preferred_key]

        if array.ndim != expected_dimensions:
            raise ValueError(
                f"{preferred_key} has {array.ndim} dimensions; "
                f"expected {expected_dimensions}."
            )

        return preferred_key, array

    candidates = {
        key: value
        for key, value in contents.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == expected_dimensions
    }

    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify one {expected_dimensions}D array "
            f"inside {path.name}. Candidates: {list(candidates)}"
        )

    return next(iter(candidates.items()))


def main() -> None:
    print("=" * 72)
    print("SALINAS HYPERSPECTRAL DATASET VERIFICATION")
    print("=" * 72)

    cube_md5 = calculate_md5(CUBE_PATH)
    label_md5 = calculate_md5(LABEL_PATH)

    if cube_md5 != EXPECTED_CUBE_MD5:
        raise ValueError(
            "The hyperspectral cube failed MD5 verification."
        )

    if label_md5 != EXPECTED_LABEL_MD5:
        raise ValueError(
            "The ground-truth file failed MD5 verification."
        )

    cube_key, cube = load_expected_array(
        CUBE_PATH,
        preferred_key="salinas_corrected",
        expected_dimensions=3,
    )

    label_key, labels = load_expected_array(
        LABEL_PATH,
        preferred_key="salinas_gt",
        expected_dimensions=2,
    )

    if cube.shape[:2] != labels.shape:
        raise ValueError(
            f"Spatial mismatch: cube {cube.shape[:2]} "
            f"versus labels {labels.shape}."
        )

    number_of_bands = cube.shape[2]

    if number_of_bands != 204:
        raise ValueError(
            f"Expected 204 spectral bands, found {number_of_bands}."
        )

    if not np.issubdtype(cube.dtype, np.number):
        raise TypeError("The hyperspectral cube is not numerical.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("The ground-truth labels are not integers.")

    if not np.isfinite(cube).all():
        raise ValueError("The hyperspectral cube contains NaN or infinity.")

    labelled_mask = labels > 0
    labelled_pixels = int(labelled_mask.sum())

    class_ids, class_counts = np.unique(
        labels[labelled_mask],
        return_counts=True,
    )

    print(f"Cube file:          {CUBE_PATH.name}")
    print(f"Cube MD5:           {cube_md5}")
    print(f"Cube variable:      {cube_key}")
    print(f"Cube shape:         {cube.shape}")
    print(f"Cube dtype:         {cube.dtype}")
    print()
    print(f"Label file:         {LABEL_PATH.name}")
    print(f"Label MD5:          {label_md5}")
    print(f"Label variable:     {label_key}")
    print(f"Label shape:        {labels.shape}")
    print(f"Label dtype:        {labels.dtype}")
    print()
    print(f"Spectral bands:     {number_of_bands}")
    print(f"Total pixels:       {labels.size:,}")
    print(f"Labelled pixels:    {labelled_pixels:,}")
    print(f"Background pixels:  {(labels == 0).sum():,}")
    print(f"Number of classes:  {len(class_ids)}")
    print(f"Minimum cube value: {cube.min()}")
    print(f"Maximum cube value: {cube.max()}")

    print("\nCLASS DISTRIBUTION")
    print("-" * 32)

    for class_id, class_count in zip(class_ids, class_counts):
        print(
            f"Class {int(class_id):2d}: "
            f"{int(class_count):6,} labelled pixels"
        )

    print("\nSALINAS DATASET TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
