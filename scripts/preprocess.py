import json
import pandas as pd
import os
import argparse

def load_data(data_path, tables_path):
    print(f"Loading data from {data_path} and {tables_path}...")
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    with open(tables_path, 'r') as f:
        tables = {}
        for line in f:
            obj = json.loads(line)
            tables[obj['id']] = obj
            
    return data, tables

def to_sql_string(sql_json, table):
    """
    Convert WikiSQL representation to SQL string.
    sql_json: {'sel': index, 'agg': index, 'conds': [[col_index, op_index, val], ...]}
    table: table schema dictionary
    """
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP', 'IS'] # WikiSQL ops often [=, >, <, OP]?? Actually, let's verify standard WikiSQL ops.
    # Standard WikiSQL ops: ['=', '>', '<', 'OP'] but let's stick to standard SQL.
    # Usually: 0: =, 1: >, 2: <
    
    # Let's perform a safe lookup
    COND_OPS = ['=', '>', '<'] 
    
    header = table['header']
    
    # SELECT clause
    sel_col = header[sql_json['sel']]
    agg = agg_ops[sql_json['agg']]
    
    if agg:
        select_part = f"SELECT {agg}({sel_col})"
    else:
        select_part = f"SELECT {sel_col}"
        
    # WHERE clause
    where_parts = []
    for cond in sql_json['conds']:
        # cond is [column_index, operator_index, comparison_value]
        col_idx, op_idx, val = cond
        output_val = f'"{val}"' if isinstance(val, str) else val
        cond_str = f"{header[col_idx]} {COND_OPS[op_idx]} {output_val}"
        where_parts.append(cond_str)
        
    if where_parts:
        where_part = " WHERE " + " AND ".join(where_parts)
    else:
        where_part = ""
        
    return select_part + where_part + f" FROM {table['id']}" # Using table ID as table name for now

def preprocess(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'dev', 'test']:
        data_file = os.path.join(data_dir, f'{split}.jsonl')
        tables_file = os.path.join(data_dir, 'tables.jsonl') # Logic usually matches split or is a master table file
        # WikiSQL usually shares tables.jsonl for train/dev, but test might have its own? 
        # Actually usually it's splitted.
        # But commonly we just use the split's jsonl. The instructions say "tables.jsonl" is placed in data/.
        
        if not os.path.exists(data_file):
            print(f"Skipping {split}, file not found.")
            continue
            
        print(f"Processing {split}...")
        
        # Load
        # Note: In WikiSQL, tables.jsonl is often one separate file or per-split.
        # We will assume 'tables.jsonl' is in data/ and contains all or we stick to the user instruction "Place ... tables.jsonl in data/".
        
        data_list, tables_dict = load_data(data_file, os.path.join(data_dir, 'tables.jsonl'))
        
        processed_examples = []
        
        for item in data_list:
            question = item['question']
            table_id = item['table_id']
            table = tables_dict.get(table_id)
            
            if not table:
                continue
                
            # Create Schema String for Input
            # Format: "header1 : type1, header2 : type2, ..."
            headers = table['header']
            types = table['types']
            schema_str = ", ".join([f"{h} : {t}" for h, t in zip(headers, types)])
            
            input_text = f"translate English to SQL: {question} Table: {table_id}, Schema: {schema_str}"
            
            # Create Target SQL
            target_sql = to_sql_string(item['sql'], table)
            
            processed_examples.append({
                'input_text': input_text,
                'target_text': target_sql,
                'table_id': table_id
            })
            
        # Save
        df = pd.DataFrame(processed_examples)
        output_path = os.path.join(output_dir, f'{split}_processed.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.output_dir)
