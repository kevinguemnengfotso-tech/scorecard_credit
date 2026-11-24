"""Data loading utilities for the credit scorecard project."""

import os
import sqlite3
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


def load_datas(raw_path: str, db_path: str, sql_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw CSV, create SQLite DB via SQLAlchemy, return full DataFrame.

    The CSV is the source of truth. The DB is created as an intermediate
    artifact for SQL-based exploration. All columns are returned (no filtering)
    so the pipeline has access to every feature.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    db_path : str
        Path to the SQLite database to create.
    sql_path : str, optional
        Path to a SQL schema file. If None, a default schema is used.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with all columns renamed and target created.
    """
    # 1. Load CSV
    df = pd.read_csv(raw_path)

    # 2. Drop useless index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 3. Rename columns for readability
    column_mapping = {
        "Attribute1": "checking_status",
        "Attribute2": "duration",
        "Attribute3": "credit_history",
        "Attribute4": "purpose",
        "Attribute5": "credit_amount",
        "Attribute6": "savings",
        "Attribute7": "employment",
        "Attribute8": "installment_rate",
        "Attribute9": "personal_status_sex",
        "Attribute10": "other_debtors",
        "Attribute11": "residence_since",
        "Attribute12": "property",
        "Attribute13": "age",
        "Attribute14": "other_installment_plans",
        "Attribute15": "housing",
        "Attribute16": "existing_credits",
        "Attribute17": "job",
        "Attribute18": "num_dependents",
        "Attribute19": "telephone",
        "Attribute20": "foreign_worker",
        "class": "class",
    }
    df = df.rename(columns=column_mapping)

    # 4. Create target variable (0 = Good, 1 = Bad/Default)
    df["res"] = (df["class"] == 2).astype(int)

    # 5. Drop original 'class' column
    df = df.drop(columns=["class"])

    # 6. Create SQLite DB via SQLAlchemy
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    # 7. Write DataFrame to DB
    df.to_sql("clients", engine, if_exists="replace", index=False)

    # 8. Create indexes for performance
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_age ON clients(age)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_res ON clients(res)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_checking ON clients(checking_status)"))
        conn.commit()

    # 9. Read ALL columns from DB (no filtering)
    result = pd.read_sql_query("SELECT * FROM clients", engine)
    engine.dispose()

    return result


# Alias for backward compatibility (tests use load_data)
load_data = load_datas
