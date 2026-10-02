"""Official SFBench/SF2Bench data loader.

The loader follows the directory and split conventions from AslanDing/SFBench.
It does not alter the official data layout:

    Processed_hour/{WATER,WELL,RAIN,PUMP,GATE}/S_*/<station>/*.csv

Each sample contains heterogeneous node time series and predicts WATER nodes.
Normalization statistics are fitted on the train phase only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


CATEGORIES = ("WATER", "RAIN", "WELL", "PUMP", "GATE")
SPLITS = ("S_5", "S_6", "S_7")
PARTITION_FILE_INDEX = {"S_5": 5, "S_6": 6, "S_7": 7}

SPLIT_DATES: Mapping[str, Mapping[str, Tuple[str, str]]] = {
    "S_5": {
        "train": ("2010-01-01 00:00:00", "2012-12-31 23:59:59"),
        "val": ("2013-01-01 00:00:00", "2013-12-31 23:59:59"),
        "test": ("2014-01-01 00:00:00", "2014-12-31 23:59:59"),
    },
    "S_6": {
        "train": ("2015-01-01 00:00:00", "2017-12-31 23:59:59"),
        "val": ("2018-01-01 00:00:00", "2018-12-31 23:59:59"),
        "test": ("2019-01-01 00:00:00", "2019-12-31 23:59:59"),
    },
    "S_7": {
        "train": ("2020-01-01 00:00:00", "2021-12-31 23:59:59"),
        "val": ("2022-01-01 00:00:00", "2022-12-31 23:59:59"),
        "test": ("2023-01-01 00:00:00", "2023-12-31 23:59:59"),
    },
}

TIME_ALIASES = {
    "0H": 0,
    "1H": 1,
    "6H": 6,
    "12H": 12,
    "1D": 24,
    "2D": 48,
    "3D": 72,
    "4D": 96,
    "5D": 120,
    "7D": 168,
    "8D": 192,
    "10D": 240,
    "1W": 168,
}


@dataclass(frozen=True)
class SFBenchMetadata:
    """Static metadata shared by train/val/test datasets."""

    station_names: List[str]
    station_categories: List[str]
    node_type: torch.Tensor
    coords: torch.Tensor
    water_indices: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return len(self.station_names)

    @property
    def num_water_nodes(self) -> int:
        return int(self.water_indices.numel())


def parse_duration(value: str) -> int:
    if value not in TIME_ALIASES:
        raise ValueError(f"Unsupported duration {value!r}; expected one of {sorted(TIME_ALIASES)}")
    return TIME_ALIASES[value]


def resolve_processed_hour_root(dataset_root: str | Path) -> Path:
    """Resolve a root that may be either the dataset folder or Processed_hour."""

    root = Path(dataset_root).expanduser().resolve()
    candidates = [
        root,
        root / "Processed_hour",
        root / "dataset" / "Processed_hour",
        root / "SFBench" / "Processed_hour",
        root / "SFBench" / "dataset" / "Processed_hour",
    ]
    for candidate in candidates:
        if all((candidate / cat).is_dir() for cat in CATEGORIES):
            return candidate
    raise FileNotFoundError(
        "Could not find official SFBench Processed_hour directory under "
        f"{root}. Expected subdirectories: {', '.join(CATEGORIES)}."
    )


def _read_station_location(json_path: Path) -> Tuple[float, float]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    x = payload.get("X COORD", payload.get("Longitude", payload.get("longitude", 0.0)))
    y = payload.get("Y COORD", payload.get("Latitude", payload.get("latitude", 0.0)))
    return float(x), float(y)


def _find_official_aux_file(processed_root: Path, filename: str) -> Path:
    # The release ships the official three-part station maps separately from
    # the large data archive so they remain versioned with the code.
    release_root = Path(__file__).resolve().parents[2]
    candidates = [
        processed_root / filename,
        processed_root.parent / filename,
        processed_root.parent / "dataset" / filename,
        release_root / "data" / "partitions" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing official SFBench auxiliary file {filename!r}. "
        "Checked the data root and this release's data/partitions directory."
    )


def _find_station_files(category_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    stations: Dict[str, Tuple[Path, Path]] = {}
    for station_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
        csv_files = sorted(station_dir.glob("*.csv"))
        json_files = sorted(station_dir.glob("*.json"))
        if not csv_files or not json_files:
            continue
        stations[station_dir.name] = (csv_files[0], json_files[0])
    return stations


def _load_category_frame(
    processed_root: Path,
    split: str,
    category: str,
    phase_dates: Tuple[str, str],
    selected_names: Optional[Iterable[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    category_dir = processed_root / category / split
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Missing category directory: {category_dir}")

    selected = set(selected_names) if selected_names is not None else None
    station_files = _find_station_files(category_dir)
    if selected is not None:
        missing = sorted(selected - set(station_files))
        if missing:
            raise FileNotFoundError(
                f"Official partition references {len(missing)} missing {category} stations in {split}: "
                + ", ".join(missing[:10])
            )
    names: List[str] = []
    values: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    coords: List[Tuple[float, float]] = []
    timestamps: Optional[List[pd.Timestamp]] = None

    start, end = pd.to_datetime(phase_dates[0]), pd.to_datetime(phase_dates[1])
    expected_timestamps = list(pd.date_range(start, end, freq="h"))
    for station_name, (csv_path, json_path) in station_files.items():
        if selected is not None and station_name not in selected:
            continue
        frame = pd.read_csv(csv_path)
        if "INTERPOLATED_VALUE" not in frame.columns:
            raise ValueError(f"{csv_path} missing INTERPOLATED_VALUE column")
        frame["TIMESTAMP_"] = pd.to_datetime(frame["TIMESTAMP"])
        frame = frame[(frame["TIMESTAMP_"] >= start) & (frame["TIMESTAMP_"] <= end)]
        if frame.empty:
            continue
        series = frame["INTERPOLATED_VALUE"].astype("float32").to_numpy()
        if np.isnan(series).any():
            continue
        confidence = frame.get("CONFIDENCE", pd.Series(np.ones(len(frame), dtype=np.float32)))
        mask = (confidence.astype("float32").to_numpy() > 0).astype("float32")
        current_timestamps = list(frame["TIMESTAMP_"])
        if current_timestamps != expected_timestamps:
            raise ValueError(
                f"Station {station_name} does not match the complete hourly phase grid in {category}/{split}"
            )
        if timestamps is None:
            timestamps = current_timestamps
        elif current_timestamps != timestamps:
            raise ValueError(f"Station {station_name} has inconsistent timestamps in {category}/{split}")
        names.append(station_name)
        values.append(series)
        masks.append(mask)
        coords.append(_read_station_location(json_path))

    if not names:
        raise ValueError(f"No usable stations found for {category}/{split}")

    return (
        names,
        np.stack(values).astype("float32"),
        np.stack(masks).astype("float32"),
        np.asarray(coords, dtype="float32"),
        timestamps or [],
    )


def _load_split_arrays_uncached(
    processed_root: Path,
    split: str,
    phase: str,
    part: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, np.ndarray]]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}")
    if phase not in ("train", "val", "test"):
        raise ValueError("phase must be train, val, or test")

    selected_by_category: Dict[str, Optional[Iterable[str]]] = {cat: None for cat in CATEGORIES}
    if part is not None:
        part_file = _find_official_aux_file(
            processed_root, f"threeparts_{part}_map_locations_{PARTITION_FILE_INDEX[split]}.json"
        )
        with part_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        selected_by_category = {cat: payload.get(cat, []) for cat in CATEGORIES}

    values: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    names: Dict[str, List[str]] = {}
    coords: Dict[str, np.ndarray] = {}
    reference_timestamps: Optional[List[pd.Timestamp]] = None
    for category in CATEGORIES:
        selected = selected_by_category.get(category)
        if selected is not None and len(list(selected)) == 0:
            continue
        n, v, m, c, category_timestamps = _load_category_frame(
            processed_root, split, category, SPLIT_DATES[split][phase], selected
        )
        if reference_timestamps is None:
            reference_timestamps = category_timestamps
        elif category_timestamps != reference_timestamps:
            raise ValueError(f"Timestamp values differ across categories in {split}/{phase}")
        values[category] = v
        masks[category] = m
        names[category] = n
        coords[category] = c
    return values, masks, names, coords


def load_split_arrays(
    processed_root: Path,
    split: str,
    phase: str,
    part: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, np.ndarray]]:
    """Load one phase, using a local binary cache after CSV validation.

    A full experiment matrix otherwise reparses the same station CSV files for
    every repeated experiment and model. The cache is stored beside the downloaded data (and
    is therefore excluded from Git), keyed by split, phase, and spatial part.
    """

    cache_dir = processed_root.parent / ".cff_cache_v4"
    part_label = "all" if part is None else str(part)
    cache_path = cache_dir / f"{split}_{phase}_part_{part_label}.npz"
    if cache_path.exists():
        archive = np.load(cache_path, allow_pickle=False)
        categories = [str(value) for value in archive["categories"].tolist()]
        values = {category: archive[f"values_{category}"] for category in categories}
        masks = {category: archive[f"masks_{category}"] for category in categories}
        names = {category: [str(value) for value in archive[f"names_{category}"].tolist()] for category in categories}
        coords = {category: archive[f"coords_{category}"] for category in categories}
        return values, masks, names, coords

    values, masks, names, coords = _load_split_arrays_uncached(processed_root, split, phase, part)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {"categories": np.asarray(list(values), dtype="U16")}
    for category in values:
        payload[f"values_{category}"] = values[category]
        payload[f"masks_{category}"] = masks[category]
        payload[f"names_{category}"] = np.asarray(names[category], dtype="U128")
        payload[f"coords_{category}"] = coords[category]
    # Multiple experiment workers may populate the same cold cache. Publish a
    # fully written archive atomically so readers never observe a partial zip.
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=cache_path.stem + "_", suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez(temporary_path, **payload)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return values, masks, names, coords


def _concat_by_category(
    values: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    names: Mapping[str, List[str]],
    coords: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    value_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    station_names: List[str] = []
    station_categories: List[str] = []
    coord_parts: List[np.ndarray] = []
    type_parts: List[np.ndarray] = []
    for type_idx, category in enumerate(CATEGORIES):
        if category not in values:
            continue
        value_parts.append(values[category])
        mask_parts.append(masks[category])
        station_names.extend(names[category])
        station_categories.extend([category] * len(names[category]))
        coord_parts.append(coords[category])
        type_parts.append(np.full((values[category].shape[0],), type_idx, dtype="int64"))
    return (
        np.concatenate(value_parts, axis=0),
        np.concatenate(mask_parts, axis=0),
        station_names,
        station_categories,
        np.concatenate(coord_parts, axis=0),
        np.concatenate(type_parts, axis=0),
    )


def fit_metadata(
    processed_root: Path,
    split: str,
    part: Optional[int],
    eps: float = 1e-6,
) -> SFBenchMetadata:
    train_values, train_masks, train_names, train_coords = load_split_arrays(processed_root, split, "train", part)
    values, masks, station_names, station_categories, coords, node_type = _concat_by_category(
        train_values, train_masks, train_names, train_coords
    )
    # Fit statistics on valid training observations only.  This keeps the
    # normalization contract aligned with the masked loss and prevents
    # invalid/interpolated entries from changing the scale of a station.
    valid_count = np.maximum(masks.sum(axis=1, keepdims=True), 1.0)
    mean = (values * masks).sum(axis=1, keepdims=True) / valid_count
    centered = (values - mean) * masks
    std = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / valid_count)
    std = np.where(std < eps, 1.0, std)
    water_count = train_values["WATER"].shape[0]
    return SFBenchMetadata(
        station_names=station_names,
        station_categories=station_categories,
        node_type=torch.from_numpy(node_type).long(),
        coords=torch.from_numpy(coords).float(),
        water_indices=torch.arange(water_count, dtype=torch.long),
        mean=torch.from_numpy(mean.astype("float32")),
        std=torch.from_numpy(std.astype("float32")),
    )


class SFBenchDataset(Dataset):
    """Windowed SFBench dataset using train-fitted metadata."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        phase: str,
        lookback: str = "2D",
        horizon: str = "1D",
        span: str = "0H",
        part: Optional[int] = None,
        metadata: Optional[SFBenchMetadata] = None,
        norm_clip: float = 0.0,
    ) -> None:
        self.processed_root = resolve_processed_hour_root(processed_root)
        self.split = split
        self.phase = phase
        self.lookback = parse_duration(lookback)
        self.horizon = parse_duration(horizon)
        self.span = parse_duration(span)
        self.part = part
        self.norm_clip = float(norm_clip)
        self.metadata = metadata or fit_metadata(self.processed_root, split, part)
        phase_start, phase_end = SPLIT_DATES[split][phase]

        values_by_cat, masks_by_cat, names_by_cat, coords_by_cat = load_split_arrays(
            self.processed_root, split, phase, part
        )
        values, masks, station_names, station_categories, _coords, _node_type = _concat_by_category(
            values_by_cat, masks_by_cat, names_by_cat, coords_by_cat
        )
        if station_names != self.metadata.station_names:
            raise ValueError(
                "Station ordering differs from train metadata. "
                "This usually means official split folders differ across phases."
            )
        self.values = torch.from_numpy(values.astype("float32"))
        self.masks = torch.from_numpy(masks.astype("float32"))
        self.normalized_values = (self.values - self.metadata.mean) / self.metadata.std
        if self.norm_clip > 0:
            self.normalized_values = self.normalized_values.clamp(-self.norm_clip, self.norm_clip)
        self.normalized_values = self.normalized_values * self.masks
        self.length = self.values.shape[1] - self.lookback - self.span - self.horizon + 1
        self.timestamps = list(pd.date_range(phase_start, phase_end, freq="h"))
        if len(self.timestamps) != self.values.shape[1]:
            raise ValueError("Loaded series length does not match the expected hourly phase range")
        if self.length <= 0:
            raise ValueError("Time window is longer than phase time series.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_start = idx
        in_end = idx + self.lookback
        out_start = in_end + self.span
        out_end = out_start + self.horizon
        water_idx = self.metadata.water_indices
        return {
            "x": self.normalized_values[:, in_start:in_end],
            "x_mask": self.masks[:, in_start:in_end],
            "y": self.normalized_values[water_idx, out_start:out_end],
            "y_mask": self.masks[water_idx, out_start:out_end],
            "y_raw": self.values[water_idx, out_start:out_end],
        }

    def denormalize_water(self, y_norm: torch.Tensor) -> torch.Tensor:
        water_idx = self.metadata.water_indices
        mean = self.metadata.mean[water_idx].to(y_norm.device)
        std = self.metadata.std[water_idx].to(y_norm.device)
        return y_norm * std.unsqueeze(0) + mean.unsqueeze(0)


class SFBenchDataModule:
    """Small data module with official train/val/test phases."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "S_6",
        lookback: str = "2D",
        horizon: str = "1D",
        span: str = "0H",
        part: Optional[int] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        norm_clip: float = 0.0,
    ) -> None:
        self.processed_root = resolve_processed_hour_root(dataset_root)
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.span = span
        self.part = part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm_clip = float(norm_clip)
        self.metadata = fit_metadata(self.processed_root, split, part)
        self.train = SFBenchDataset(
            self.processed_root, split, "train", lookback, horizon, span, part, self.metadata, self.norm_clip
        )
        self.val = SFBenchDataset(
            self.processed_root, split, "val", lookback, horizon, span, part, self.metadata, self.norm_clip
        )
        self.test = SFBenchDataset(
            self.processed_root, split, "test", lookback, horizon, span, part, self.metadata, self.norm_clip
        )

    def loader(self, phase: str, shuffle: Optional[bool] = None, drop_last: bool = False) -> DataLoader:
        dataset = {"train": self.train, "val": self.val, "test": self.test}[phase]
        if shuffle is None:
            shuffle = phase == "train"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )
