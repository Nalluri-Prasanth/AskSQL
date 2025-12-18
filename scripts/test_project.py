import unittest
from inference import SQLGenerator
from database import execute_query, init_database
import os

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
        # Ensure DB has data (dummy data puts something in table_0)
        sql = "SELECT * FROM \"table_0\""
        df, err = execute_query(sql)
        self.assertIsNone(err)
        self.assertFalse(df.empty)

if __name__ == '__main__':
    unittest.main()
