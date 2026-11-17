import pandas as pd
import sqlite3
import os

def load_datas(raw_path: str, db_path: str,sql_path:str) -> pd.DataFrame:
    # 1. Charger le CSV
    df = pd.read_csv(raw_path)
    
    # 2. Supprimer l'index inutile
    df = df.drop(columns=['Unnamed: 0'])
    
    # 3. Renommer les colonnes pour la lisibilité
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
    #rename the columns
    df = df.rename(columns=column_mapping)
    
    # 4. Créer la variable cible (0 = Good, 1 = Bad/Default)
    df['res'] = (df['class'] == 2).astype(int)
    
    # 5. Supprimer la colonne 'class' originale (remplacée par 'default')
    df = df.drop(columns=['class'])
    
    # 6. Vérifier les types
    # (les catégorielles sont déjà en string, les numériques en int)
    
    # 7. Créer la base SQLite
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    #link to the sql
    conn = sqlite3.connect(db_path)
    
    # 8. Lire le schéma SQL
    with open(sql_path, 'r') as f:
        schema = f.read()
        print(schema)
    conn.executescript(schema)
    
    # 9. Insérer les données
    df.to_sql('clients', conn, if_exists='replace', index=False)
    
    # 10. Créer des index pour la performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_age ON clients(age)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_res ON clients(res)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_checking ON clients(checking_status)')
    conn.commit()
    
    # 11. Lire les données avec une requête SQL 
    query = """
SELECT 
    c.age, c.credit_amount, c.employment, c.housing,
    c.duration, c.installment_rate, c.property,
    c.res
FROM clients c
WHERE c.credit_amount > 0
"""
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    return result