"""Download and prepare the ANN-Benchmarks datasets used by this project."""

import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import h5py
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent / "data"
DATASETS = {
    "glove-200-angular": (
        "https://ann-benchmarks.com/glove-200-angular.hdf5",
        962_819_488,
    ),
    "coco-i2i-512": (
        "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/"
        "coco-i2i-512-angular.hdf5",
        142_254_080,
    ),
}
BATCH_SIZE = 8_192

def get_source(directory, url, size):
    path = directory / "source.hdf5"
    if path.is_file() and path.stat().st_size == size:
        return path

    print(f"Downloading {directory.name} ({size / 2**20:.0f} MiB)...", flush=True)
    with TemporaryDirectory(dir=directory) as work_dir:
        partial = Path(work_dir) / path.name
        with urlopen(url, timeout=60) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=2**20)
        if partial.stat().st_size != size or not h5py.is_hdf5(partial):
            raise ValueError(f"{directory.name}: download has an unexpected size or format")
        partial.replace(path)
    return path


def export_normalized(vectors, path):
    with path.open("wb") as output:
        for start in range(0, len(vectors), BATCH_SIZE):
            batch = np.asarray(vectors[start:start + BATCH_SIZE], dtype="<f4")
            batch_norms = np.linalg.norm(batch, axis=1)
            if not np.isfinite(batch_norms).all() or (batch_norms <= 0).any():
                raise ValueError(f"{vectors.name}: nonfinite or zero norm in batch at row {start}")
            batch /= batch_norms[:, None]
            batch.tofile(output)


def prepare(name, url, size):
    destination = DATA_ROOT / name
    destination.mkdir(parents=True, exist_ok=True)
    source = get_source(destination, url, size)
    print(f"Preparing {name}...", flush=True)

    with TemporaryDirectory(dir=destination) as work_dir, h5py.File(source, "r") as dataset:
        work = Path(work_dir)
        database, queries = dataset["train"], dataset["test"]
        if (database.ndim != 2 or queries.ndim != 2
                or not all(database.shape + queries.shape)
                or database.shape[1] != queries.shape[1]):
            raise ValueError(f"{name}: expected nonempty vector matrices with matching dimensions")

        neighbors = dataset["neighbors"][:]
        if (neighbors.ndim != 2 or neighbors.shape[0] != queries.shape[0]
                or neighbors.shape[1] == 0):
            raise ValueError(f"{name}: expected one nonempty neighbor row per query")
        max_id = min(database.shape[0] - 1, np.iinfo(np.int32).max)
        if (not np.issubdtype(neighbors.dtype, np.integer)
                or neighbors.min() < 0 or neighbors.max() > max_id):
            raise ValueError(f"{name}: neighbor IDs must be valid int32 database row numbers")

        export_normalized(database, work / "database.f32")
        export_normalized(queries, work / "queries.f32")
        neighbors.astype("<i4").tofile(work / "neighbors.i32")
        metadata = {
            "database_rows": database.shape[0],
            "query_rows": queries.shape[0],
            "dimension": database.shape[1],
            "neighbors_per_query": neighbors.shape[1],
        }
        (work / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for filename in ("database.f32", "queries.f32", "neighbors.i32", "metadata.json"):
            (work / filename).replace(destination / filename)


def main():
    try:
        for name, (url, size) in DATASETS.items():
            prepare(name, url, size)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
