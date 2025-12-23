# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from math import floor
from pathlib import Path

import datasets
import fire
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds
import ray
from audio_tools import AudioTableProcessor, map_to_target_schema
from datasets import load_dataset
from text_tools import text_normalize

class TarsilaTextProcessor:
    """
    Batch-level processor for Tarsila text data processing.

    Handles digit replacement, text normalization, and language mapping
    for batches of FLEURS data.
    """

    def __init__(self):
        """
        Initialize the processor.
        lang: por_Latn
        """

    def __call__(self, batch: pa.Table) -> pa.Table:
        # Extract transcription column as Python list
        transcriptions = batch["text"].to_pylist()
        processed_transcriptions = []

        for text in transcriptions:
            # Normalize text
            processed_text = text_normalize(text, iso_code="pt")
            processed_transcriptions.append(processed_text)

        # add the processed transcription
        batch = batch.append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )

        language_values = ["por_Latn"] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )

        return batch


class DataPrepCLI:
    """Command-line interface for ASR data preparation tasks."""

    @staticmethod
    def check_versions():
        """Check and display versions of critical packages used in data preparation.

        This helps ensure compatibility and reproducibility of the data preparation pipeline.
        """
        print("📦 Package Versions:")
        print(f"  datasets: {datasets.__version__}")
        print(f"  pyarrow:  {pa.__version__}")
        print(f"  ray:      {ray.__version__}")
        print(f"  polars:   {pl.__version__}")

        # Check for known compatibility issues
        if hasattr(datasets, "__version__"):
            datasets_ver = tuple(map(int, datasets.__version__.split(".")))
            if datasets_ver >= (3, 6, 0):
                print(
                    "⚠️  Warning: datasets version >= 3.6.0 may have compatibility issues"
                )

        if hasattr(ray, "__version__"):
            ray_ver = tuple(
                map(int, ray.__version__.split(".")[:2])
            )  # Major.minor only
            if ray_ver < (2, 49):
                print("⚠️  Warning: ray version < 2.49 may have performance issues")

    def _ingest_tarsila_internal(
        self, output_dir: str | None = None
    ):
        """Internal method for Tarsila ingestion."""
        # see https://huggingface.co/datasets/sidleal/TARSILA-ASR-V1

        split_renaming = {"validation": "dev"}

        for split in ["test", "validation", "train"]:
            tarsila_hf = load_dataset(
                "sidleal/TARSILA-ASR-V1",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            tarsila_hf = tarsila_hf.shuffle(seed=123, buffer_size=10000)
            ray_ds_stream_ = ray.data.from_huggingface(tarsila_hf)

            # Use batch-level text processing
            num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)
            ray_ds_stream_ = ray_ds_stream_.map_batches(
                TarsilaTextProcessor,
                fn_constructor_kwargs={},
                batch_size=1000,
                batch_format="pyarrow",
                concurrency=num_cpus,
            )

            # Audio processing
            ray_ds_stream_ = ray_ds_stream_.map_batches(
                AudioTableProcessor,
                fn_constructor_kwargs={
                    "audio_column": "audio.bytes",
                    "audio_format": "flac",  # or "ogg", "wav", etc.
                },
                batch_size=100,
                batch_format="pyarrow",
                concurrency=num_cpus,
            )
            ray_ds_stream_ = ray_ds_stream_.map_batches(
                partial(
                    map_to_target_schema,
                    split=split_renaming.get(split, split),
                    corpus="fleurs",
                ),
                batch_size=100,
                batch_format="pyarrow",
            )
            ray_ds_stream_.write_parquet(
                output_dir,
                partition_cols=["corpus", "split", "language"],
                min_rows_per_file=10_000,
                row_group_size=100,  # https://github.com/ray-project/ray/issues/52481
            )

    @staticmethod
    def _compute_distribution_stats_internal(
        parquet_dataset_root: str, output_path: str
    ):
        """Internal method for computing distribution statistics."""
        table = pa_ds.dataset(
            parquet_dataset_root, partitioning="hive", exclude_invalid_files=True
        ).to_table(columns=["language", "corpus", "audio_size"])
        pl_table = pl.from_arrow(table.combine_chunks())
        assert isinstance(pl_table, pl.DataFrame)
        stats = pl_table.group_by(["corpus", "language"]).agg(
            (pl.col("audio_size").sum() / 3600 / 16_000).alias("hours")
        )
        stats.write_csv(output_path, separator="\t")
        return output_path

    def ingest_tarsila(self, output_dir: str):
        """Ingest Tarsila dataset.

        Args:
            output_dir: Output directory path for processed Parquet files
        """
        print(f"Starting Tarsila ingestion to: {output_dir}")
        self._ingest_tarsila_internal(output_dir)
        print("Tarsila ingestion completed")

    def compute_stats(self, parquet_dataset_root: str, output_path: str):
        """Compute distribution statistics from processed datasets.

        Args:
            parquet_dataset_root: Path to the root of partitioned Parquet dataset
            output_path: Output path for TSV statistics file
        """
        print(f"Computing stats for: {parquet_dataset_root}")
        result_path = self._compute_distribution_stats_internal(
            parquet_dataset_root, output_path
        )
        print(f"Statistics saved to: {result_path}")
        return result_path

    def test_dataset(self, dataset_path: str, **kwargs):
        """
        Test dataset functionality - redirects to dedicated dataloader_example module.

        Args:
            dataset_path: Path to the dataset directory
            **kwargs: Additional arguments passed to dataloader_example
        """
        print("📚 For dataset testing, use the dedicated dataloader_example module:")
        print(
            f"   python -m workflows.dataprep.dataloader_example test_dataset --dataset_path='{dataset_path}'"
        )
        print(
            f"   or: python -m workflows.dataprep.dataloader_example --dataset_path='{dataset_path}' --split='train' --num_iterations=10 - reset"
        )
        print("\n🔧 Available method:")
        print("   • test_dataset: Basic dataset testing with iterations")

        from dataloader_example import DataLoaderExample

        loader = DataLoaderExample()
        return loader.test_dataset(dataset_path, **kwargs)

    def run(
        self, output_dir: str, name: str = "tarsila", version: str = "1"
    ):
        """Run data preparation pipeline for Tarsila ASR

        Args:
            output_dir: Base output directory path
            name: Dataset name (default: "tarsila")
            version: Dataset version (default: "1")
        """
        print("🚀 Starting SHORT data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        print("🔄 Ingesting Tarsila dataset...")
        self._ingest_tarsila_internal(
            parquet_dataset_root
        )

        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ SHORT pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path, num_iterations=5)
        return parquet_dataset_root, stats_path

if __name__ == "__main__":
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    try:
        fire.Fire(DataPrepCLI)
    finally:
        # Clean shutdown of Ray
        if ray.is_initialized():
            ray.shutdown()
