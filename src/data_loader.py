import pandas as pd
import os
from sqlalchemy import create_engine, text

def load_datas(raw_path: str, db_path: str, sql_path: str = None) -> pd.DataFrame:
    """Load raw CSV, create SQLite DB via SQLAlchemy, return full DataFrame.

    Rationale: The CSV is the source of truth. We create the DB as an
    intermediate artifact for SQL-based exploration, but return ALL columns
    (not a subset) so the pipeline has access to every feature.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file (german.data or german.csv).
    db_path : str
        Path to the SQLite database to create.
    sql_path : str, optional
        Path to a SQL schema file. If None, a default schema is used.

    Returns
    -------
    pd.DataFrame
        Full DataFrame with all columns renamed and target created.
    """
    # 1. Charger le CSV
    df = pd.read_csv(raw_path)

    # 2. Supprimer l'index inutile si present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 3. Renommer les colonnes pour la lisibilite
    column_mapping = {
        'Attribute1': 'checking_status',
        'Attribute2': 'duration',
        'Attribute3': 'credit_history',
        'Attribute4': 'purpose',
        'Attribute5': 'credit_amount',
        'Attribute6': 'savings',
        'Attribute7': 'employment',
        'Attribute8': 'installment_rate',
        'Attribute9': 'personal_status_sex',
        'Attribute10': 'other_debtors',
        'Attribute11': 'residence_since',
        'Attribute12': 'property',
        'Attribute13': 'age',
        'Attribute14': 'other_installment_plans',
        'Attribute15': 'housing',
        'Attribute16': 'existing_credits',
        'Attribute17': 'job',
        'Attribute18': 'num_dependents',
        'Attribute19': 'telephone',
        'Attribute20': 'foreign_worker',
        'class': 'class'
    }
    df = df.rename(columns=column_mapping)

    # 4. Creer la variable cible (0 = Good, 1 = Bad/Default)
    df['res'] = (df['class'] == 2).astype(int)

    # 5. Supprimer la colonne 'class' originale
    df = df.drop(columns=['class'])

    # 6. Creer la base SQLite via SQLAlchemy
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")

    # 7. Ecrire le DataFrame dans la DB
    df.to_sql('clients', engine, if_exists='replace', index=False)

    # 8. Creer des index pour la performance
    with engine.connect() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_age ON clients(age)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_res ON clients(res)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_checking ON clients(checking_status)'))
        conn.commit()

    # 9. Lire TOUTES les colonnes depuis la DB (pas de filtre)
    # Rationale: la source de verite est le CSV, la DB est un artefact.
    # On retourne le DataFrame complet pour que la pipeline ait acces
    # a toutes les features.
    result = pd.read_sql_query("SELECT * FROM clients", engine)
    engine.dispose()

    return result
