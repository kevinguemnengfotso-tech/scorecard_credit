-- Table clients
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checking_status TEXT NOT NULL,
    duration INTEGER NOT NULL,
    credit_history TEXT NOT NULL,
    purpose TEXT NOT NULL,
    credit_amount INTEGER NOT NULL,
    savings TEXT NOT NULL,
    employment TEXT NOT NULL,
    installment_rate INTEGER NOT NULL,
    personal_status_sex TEXT NOT NULL,
    other_debtors TEXT NOT NULL,
    residence_since INTEGER NOT NULL,
    property TEXT NOT NULL,
    age INTEGER NOT NULL,
    other_installment_plans TEXT NOT NULL,
    housing TEXT NOT NULL,
    existing_credits INTEGER NOT NULL,
    job TEXT NOT NULL,
    num_dependents INTEGER NOT NULL,
    telephone TEXT NOT NULL,
    foreign_worker TEXT NOT NULL,
    res INTEGER NOT NULL
);

-- Index pour la performance
CREATE INDEX IF NOT EXISTS idx_age ON clients(age);
CREATE INDEX IF NOT EXISTS idx_res ON clients(res);
CREATE INDEX IF NOT EXISTS idx_checking ON clients(checking_status);
CREATE INDEX IF NOT EXISTS idx_credit_amount ON clients(credit_amount);