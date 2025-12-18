import sqlite3
import pandas as pd
import os
import json

DB_FILE = 'project_db.sqlite'


def init_database(source_file='data/animenew.csv', is_csv=True):
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found.")
        return

    print(f"Initializing database from {source_file}...")
    
    if is_csv:
        try:
            # Read CSV
            df = pd.read_csv(source_file)
            table_id = "anime" # Default table name for the CSV
            
            # Clean column names: remove spaces, special chars
            df.columns = [c.strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in df.columns]
            
            # Write to SQLite
            df.to_sql(table_id, conn, if_exists='replace', index=False)
            print(f"Loaded {len(df)} rows into table '{table_id}'")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
    else:
        # Legacy JSONL support
        with open(source_file, 'r') as f:
            for line in f:
                table = json.loads(line)
                table_id = table['id']
                headers = table['header']
                types = table['types']
                rows = table.get('rows', [])
                
                # Map WikiSQL types to SQLite types
                type_map = {'text': 'TEXT', 'real': 'REAL', 'integer': 'INTEGER'}
                col_defs = []
                for h, t in zip(headers, types):
                    # Sanitize column names
                    h_safe = h.replace(" ", "_").replace("/", "_")
                    col_defs.append(f'"{h_safe}" {type_map.get(t, "TEXT")}')
                
                create_sql = f"CREATE TABLE IF NOT EXISTS \"{table_id}\" ({', '.join(col_defs)})"
                cursor.execute(create_sql)
                
                # Insert rows
                if rows:
                    placeholders = ', '.join(['?'] * len(headers))
                    insert_sql = f"INSERT INTO \"{table_id}\" VALUES ({placeholders})"
                    cursor.executemany(insert_sql, rows)
                
    conn.commit()
    conn.close()
    print("Database initialized.")

def execute_query(sql):
    try:
        conn = sqlite3.connect(DB_FILE)
        # return result as dataframe
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    init_database()
