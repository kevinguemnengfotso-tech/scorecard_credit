import os
import sqlite3
import pandas as pd
import pytest
from src.data_loader import load_data


@pytest.fixture(autouse=True)
def cleanup():
  """Clean the test database before and after each test."""
  db_path = "data/processed/test.db"
  if os.path.exists(db_path):
    os.remove(db_path)
  yield
  if os.path.exists(db_path):
    os.remove(db_path)


def test_load_data_shape():
  """Check that loaded data has the expected shape."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  assert df.shape[0] == 1000
  assert "res" in df.columns
  assert df["res"].isin([0, 1]).all()


def test_load_data_no_nulls():
  """Check that there are no missing values."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  assert df.isnull().sum().sum() == 0


def test_load_data_sql():
  """Check that the SQLite database is created and contains data."""
  db_path = "data/processed/test.db"
  # Load first to create the database and table
  load_data("data/raw/dataset_original.csv", db_path)

  conn = sqlite3.connect(db_path)
  count = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
  assert count == 1000
  conn.close()


def test_load_data_target_distribution():
  """Check the target distribution."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  # Check dataset proportions (700 / 300)
  assert df["res"].value_counts()[0] == 700
  assert df["res"].value_counts()[1] == 300