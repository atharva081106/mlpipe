"""Tests for data ingestion module."""

from pathlib import Path
import pytest

from mlflux.core.exceptions import DatasetError
from mlflux.data.ingestion import load_dataset


def test_load_valid_csv(sample_csv_file: Path):
    dataset = load_dataset(sample_csv_file)
    assert dataset.num_rows == 120
    assert dataset.num_cols == 6
    assert len(dataset.column_names) == 6
    assert dataset.sha256_hash is not None
    assert len(dataset.sha256_hash) == 64
    assert dataset.filename == "test_data.csv"
    assert dataset.memory_bytes > 0


def test_load_nonexistent_file():
    with pytest.raises(DatasetError) as exc_info:
        load_dataset("non_existent_path_xyz.csv")
    assert "not found" in str(exc_info.value).lower()


def test_load_non_csv_file(tmp_path: Path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("a,b,c\n1,2,3")
    with pytest.raises(DatasetError) as exc_info:
        load_dataset(txt_file)
    assert "unsupported file format" in str(exc_info.value).lower()


def test_load_empty_file(tmp_path: Path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.touch()
    with pytest.raises(DatasetError) as exc_info:
        load_dataset(empty_csv)
    assert "completely empty" in str(exc_info.value).lower()


def test_load_header_only_file(tmp_path: Path):
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("col1,col2,col3\n")
    with pytest.raises(DatasetError) as exc_info:
        load_dataset(header_only)
    assert "zero data rows" in str(exc_info.value).lower()
