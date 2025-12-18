from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.inference import SQLGenerator
from src.database import execute_query, init_database
import os

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

# Initialize Generator
gen = SQLGenerator()

# Helper to get schema
def get_schema(table_id="anime"):
    try:
        # We query the DB to get column names and types
        # This is a bit hacky but works for SQLite
        df, err = execute_query(f"PRAGMA table_info(\"{table_id}\")")
        if err or df is None or df.empty:
            return [], []
        
        headers = df['name'].tolist()
        types = df['type'].tolist()
        return headers, types
    except Exception as e:
        print(f"Schema Error: {e}")
        return [], []

@app.route('/', methods=['GET'])
def index():
    table_id = "anime"
    headers, types = get_schema(table_id)
    
    # Preview Data
    preview_df, err = execute_query(f"SELECT * FROM \"{table_id}\" LIMIT 5")
    preview_html = ""
    if not err and preview_df is not None:
        preview_html = preview_df.to_html(classes='table table-striped', index=False)
    
    return render_template('index.html', 
                           headers=headers, 
                           types=types, 
                           preview_html=preview_html)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    table_id = "anime"
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
        
    headers, types = get_schema(table_id)
    print(f"DEBUG SCHEMA: headers={headers}, types={types}")
    
    try:
        # 1. Generate SQL
        generated_sql = gen.generate_sql(question, table_id, headers, types)
        print(f"DEBUG Generated SQL: {generated_sql}")
        
        # 2. Execute SQL
        results, error = execute_query(generated_sql)
        
        if error:
            return jsonify({'sql': generated_sql, 'error': error})
        
        # Convert results to HTML or JSON
        if results.empty:
             return jsonify({'sql': generated_sql, 'result_html': "<p>No results found.</p>"})
        
        result_html = results.to_html(classes='table table-bordered table-hover', index=False)
        return jsonify({'sql': generated_sql, 'result_html': result_html})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/init_db', methods=['POST'])
def init_db_route():
    try:
        init_database()
        return jsonify({'message': 'Database re-initialized'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure DB exists
    if not os.path.exists('project_db.sqlite'):
        init_database()
    app.run(debug=True, port=5000)
