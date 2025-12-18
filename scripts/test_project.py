import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.inference import SQLGenerator
from src.database import execute_query, init_database

class TestTextToSQL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up test environment...")
        if not os.path.exists('project_db.sqlite'):
            init_database() # Ensure DB exists
        cls.generator = SQLGenerator() # Load model once

    def test_sql_generation(self):
        # Simple test
        question = "What is the name?"
        table_id = "table_0" # From dummy data
        headers = ["col1", "col2", "col3"]
        types = ["text", "real", "text"]
        
        sql = self.generator.generate_sql(question, table_id, headers, types)
        print(f"Question: {question} -> SQL: {sql}")
        self.assertTrue(len(sql) > 0)
        self.assertIn("SELECT", sql.upper())

    def test_db_execution(self):
        # Ensure DB has data (table is 'anime')
        sql = "SELECT * FROM \"anime\" LIMIT 1"
        df, err = execute_query(sql)
        self.assertIsNone(err)
        self.assertFalse(df.empty)

if __name__ == '__main__':
    unittest.main()
