from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat


SALINAS_CLASS_NAMES: tuple[str, ...] = (
    "Brocoli_green_weeds_1",
    "Brocoli_green_weeds_2",
    "Fallow",
    "Fallow_rough_plow",
    "Fallow_smooth",
    "Stubble",
    "Celery",
    "Grapes_untrained",
    "Soil_vinyard_develop",
    "Corn_senesced_green_weeds",
    "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk",
    "Lettuce_romaine_6wk",
    "Lettuce_romaine_7wk",
    "Vinyard_untrained",
    "Vinyard_vertical_trellis",
)

BEVERAGE_CLASS_NAMES: tuple[str, ...] = (
    "Papaya",
    "Coffee",
    "Pomegranate",
    "Orange",
    "Tea",
    "Wine",
    "Whisky",
    "Rum",
    "Brandy",
)


@dataclass(frozen=True)
class SpectralDataset:
    X: np.ndarray
    y: np.ndarray
    class_names: Sequence[str]
    name: str
    metadata: dict[str, object]


def _extract_mat_array(path: Path, preferred_key: str, ndim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    content = loadmat(path)
    if preferred_key in content:
        array = content[preferred_key]
        if array.ndim != ndim:
            raise ValueError(
                f"Variable '{preferred_key}' in {path.name} has {array.ndim} "
                f"dimensions; expected {ndim}."
            )
        return array

    candidates = [
        value
        for key, value in content.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == ndim
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify exactly one {ndim}D array in {path}."
        )
    return candidates[0]


def load_salinas(project_root: Path) -> SpectralDataset:
    data_dir = project_root / "data" / "raw" / "salinas"
    cube_path = data_dir / "Salinas_corrected.mat"
    label_path = data_dir / "Salinas_gt.mat"

    cube = _extract_mat_array(cube_path, "salinas_corrected", 3)
    labels = _extract_mat_array(label_path, "salinas_gt", 2)

    if cube.shape[:2] != labels.shape:
        raise ValueError(
            f"Spatial dimensions do not match: cube={cube.shape[:2]}, "
            f"labels={labels.shape}."
        )
    if cube.shape[2] != 204:
        raise ValueError(f"Expected 204 bands, found {cube.shape[2]}.")

    mask = labels > 0
    X = cube[mask].astype(np.float32, copy=False)
    y = labels[mask].astype(np.int64, copy=False) - 1

    if not np.isfinite(X).all():
        raise ValueError("Salinas contains NaN or infinite values.")

    unique = np.unique(y)
    expected = np.arange(len(SALINAS_CLASS_NAMES))
    if not np.array_equal(unique, expected):
        raise ValueError(
            f"Unexpected class IDs: {unique.tolist()} instead of "
            f"{expected.tolist()}."
        )

    return SpectralDataset(
        X=X,
        y=y,
        class_names=SALINAS_CLASS_NAMES,
        name="salinas_real_hsi",
        metadata={
            "source": "Zenodo 10.5281/zenodo.15771735",
            "cube_shape": tuple(int(v) for v in cube.shape),
            "labelled_pixels": int(mask.sum()),
            "bands": int(cube.shape[2]),
            "classes": len(SALINAS_CLASS_NAMES),
            "note": (
                "Real public HSI engineering validation; not the original "
                "beverage-stain dataset."
            ),
        },
    )


def _gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def generate_synthetic_beverage(
    samples_per_class: int = 2500,
    seed: int = 42,
) -> SpectralDataset:
    """Generate structured 204-band spectra for software validation only.

    This dataset is intentionally synthetic. It verifies that the exact
    9-class/204-band/162-band pipeline functions, but its accuracy must not be
    interpreted as a reproduction of the paper's forensic performance.
    """

    if samples_per_class < 100:
        raise ValueError("samples_per_class must be at least 100.")

    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 1000.0, 204, dtype=np.float32)
    normalized = (wavelengths - 400.0) / 600.0

    specs = (
        (0.45, 0.18, [(610, 75, 0.20), (850, 90, 0.10)], [(520, 30, 0.07)]),
        (0.16, 0.18, [(760, 100, 0.08)], [(470, 45, 0.08), (680, 60, 0.06)]),
        (0.10, 0.12, [(730, 80, 0.08)], [(530, 50, 0.09), (620, 55, 0.08)]),
        (0.50, 0.10, [(590, 60, 0.23), (880, 85, 0.08)], [(710, 35, 0.05)]),
        (0.18, 0.17, [(790, 95, 0.09)], [(500, 50, 0.07), (670, 65, 0.05)]),
        (0.08, 0.10, [(820, 100, 0.07)], [(545, 45, 0.12), (690, 55, 0.10)]),
        (0.27, 0.13, [(760, 100, 0.11)], [(575, 45, 0.05), (910, 35, 0.05)]),
        (0.29, 0.12, [(750, 100, 0.10)], [(560, 50, 0.045), (890, 40, 0.055)]),
        (0.25, 0.14, [(770, 95, 0.10)], [(590, 50, 0.055), (930, 40, 0.045)]),
    )

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for class_id, (base, slope, peaks, dips) in enumerate(specs):
        template = np.full_like(wavelengths, base, dtype=np.float32)
        template += slope * normalized

        for center, width, amplitude in peaks:
            template += amplitude * _gaussian(wavelengths, center, width)
        for center, width, amplitude in dips:
            template -= amplitude * _gaussian(wavelengths, center, width)

        template = np.clip(template, 0.01, 0.98)

        baseline = rng.normal(0.0, 0.018, size=(samples_per_class, 1))
        scale = rng.normal(1.0, 0.055, size=(samples_per_class, 1))
        tilt = rng.normal(0.0, 0.025, size=(samples_per_class, 1))
        white_noise = rng.normal(
            0.0,
            0.014,
            size=(samples_per_class, wavelengths.size),
        )
        smooth_noise = (
            white_noise
            + np.roll(white_noise, 1, axis=1)
            + np.roll(white_noise, -1, axis=1)
        ) / 3.0

        spectra = (
            scale * template[None, :]
            + baseline
            + tilt * (normalized[None, :] - 0.5)
            + smooth_noise
        )
        spectra = np.clip(spectra, 0.0, 1.0).astype(np.float32)

        all_X.append(spectra)
        all_y.append(np.full(samples_per_class, class_id, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    order = rng.permutation(len(y))

    return SpectralDataset(
        X=X[order],
        y=y[order],
        class_names=BEVERAGE_CLASS_NAMES,
        name="synthetic_beverage",
        metadata={
            "samples": int(len(y)),
            "samples_per_class": int(samples_per_class),
            "bands": 204,
            "classes": len(BEVERAGE_CLASS_NAMES),
            "wavelength_range_nm": [400.0, 1000.0],
            "seed": int(seed),
            "note": (
                "Synthetic software-validation data only; not the paper's "
                "dataset and not valid for forensic accuracy claims."
            ),
        },
    )
