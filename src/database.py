"""SQLite access for the anime table."""
import os
import sqlite3

import pandas as pd

DB_FILE = 'project_db.sqlite'
TABLE_NAME = 'anime'


def init_database(source_file='data/animenew.csv'):
    """Rebuild the SQLite database from the CSV. Any existing database file is deleted first."""
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found.")
        return

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    print(f"Initializing database from {source_file}...")
    df = pd.read_csv(source_file)
    # Clean column names: remove spaces and special characters
    df.columns = [c.strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in df.columns]

    conn = sqlite3.connect(DB_FILE)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.commit()
    finally:
        conn.close()
    print(f"Loaded {len(df)} rows into table '{TABLE_NAME}'")


def execute_query(sql):
    """Run a SQL statement and return (DataFrame, None) or (None, error message)."""
    try:
        conn = sqlite3.connect(DB_FILE)
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    init_database()
