import os
import sqlite3
import pandas as pd
import pytest
from src.data_loader import load_data


@pytest.fixture(autouse=True)
def cleanup():
  """Nettoie la base de test avant et après chaque test."""
  db_path = "data/processed/test.db"
  if os.path.exists(db_path):
    os.remove(db_path)
  yield
  if os.path.exists(db_path):
    os.remove(db_path)


def test_load_data_shape():
  """Vérifie que les données chargées ont la bonne forme."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  assert df.shape[0] == 1000
  assert "res" in df.columns
  assert df["res"].isin([0, 1]).all()


def test_load_data_no_nulls():
  """Vérifie qu'il n'y a pas de valeurs manquantes."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  assert df.isnull().sum().sum() == 0


def test_load_data_sql():
  """Vérifie que la base SQLite est créée et contient les données."""
  db_path = "data/processed/test.db"
  # On charge d'abord pour créer la base et la table
  load_data("data/raw/dataset_original.csv", db_path)

  conn = sqlite3.connect(db_path)
  count = conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
  assert count == 1000
  conn.close()


def test_load_data_target_distribution():
  """Vérifie la distribution de la target."""
  df = load_data("data/raw/dataset_original.csv", "data/processed/test.db")
  # Vérifie les proportions du dataset (700 / 300)
  assert df["res"].value_counts()[0] == 700
  assert df["res"].value_counts()[1] == 300