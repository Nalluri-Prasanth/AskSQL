import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


MODEL_PATH = 'models/avgs_t5_wikisql' # Predicted path after training
DEFAULT_MODEL = 'cssupport/t5-small-awesome-text-to-sql' # Better fallback

class SQLGenerator:

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Try to load the trained model if no path provided, otherwise use provided path
        # If loading fails, fallback to DEFAULT_MODEL
        target_path = model_path if model_path else MODEL_PATH
        
        print(f"Loading model from {target_path}...")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(target_path)
            self.model = T5ForConditionalGeneration.from_pretrained(target_path).to(self.device)
            print(f"Successfully loaded model from {target_path}")
        except (OSError, TypeError, ValueError):
            print(f"Model not found or corrupted at {target_path}, falling back to {DEFAULT_MODEL}")
            self.tokenizer = T5Tokenizer.from_pretrained(DEFAULT_MODEL)
            self.model = T5ForConditionalGeneration.from_pretrained(DEFAULT_MODEL).to(self.device)
            print(f"Successfully loaded fallback model: {DEFAULT_MODEL}")


    def generate_sql(self, question, table_id, headers, types):
        # Format input for cssupport/t5-small-awesome-text-to-sql
        # tables: CREATE TABLE table_name (col1 type1, ...) query for: question
        
        col_defs = []
        for h, t in zip(headers, types):
            # headers are already cleaned/from DB logic hopefully, but double check
            col_defs.append(f"{h} {t}")
        

        create_table_str = f"CREATE TABLE {table_id} ({', '.join(col_defs)})"
        
        # Heuristic for specific difficult patterns
        # Heuristic for specific difficult patterns
        lower_q = question.lower()
        if "story of" in lower_q or "description of" in lower_q or "plot of" in lower_q or "synopsis of" in lower_q:
            if "story of" in lower_q:
                target_title = lower_q.split("story of")[-1]
            elif "description of" in lower_q:
                target_title = lower_q.split("description of")[-1]
            elif "plot of" in lower_q:
                target_title = lower_q.split("plot of")[-1]
            elif "synopsis of" in lower_q:
                target_title = lower_q.split("synopsis of")[-1]
            
            target_title = target_title.strip().replace("?", "")
            return f"SELECT description FROM {table_id} WHERE title LIKE '%{target_title}%'"
            
        # Heuristic for "top N"
        import re
        top_match = re.search(r'top\s+(\d+)', question.lower())
        if top_match:
            limit = top_match.group(1)
            if "rated" in question.lower() or "rating" in question.lower():
                return f"SELECT title, rating FROM {table_id} ORDER BY rating DESC LIMIT {limit}"
            if "popular" in question.lower() or "votes" in question.lower():
                return f"SELECT title, votes FROM {table_id} ORDER BY votes DESC LIMIT {limit}"
            if "eps" in question.lower() or "episodes" in question.lower():
                 return f"SELECT title, eps FROM {table_id} ORDER BY eps DESC LIMIT {limit}"
            if "recent" in question.lower() or "new" in question.lower():
                 return f"SELECT title, startYr FROM {table_id} ORDER BY startYr DESC LIMIT {limit}"
        
        input_text = f"tables: {create_table_str} query for: {question}"
        print(f"DEBUG INPUT: {input_text}")
        
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, 
            max_length=128, 
            num_beams=4, 
            early_stopping=True
        )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"DEBUG RAW OUTPUT: {decoded_output}")
        generated_sql = decoded_output
        
        # Post-processing to fix common model hallucinations
        generated_sql = generated_sql.replace("rated", "rating")
        generated_sql = generated_sql.replace("ratings", "rating")
        generated_sql = generated_sql.replace("COUNT(rating)", "rating")
        
        # Robustness: Force LIKE for string comparisons to handle case variance
        generated_sql = generated_sql.replace("= '", "LIKE '")
        generated_sql = generated_sql.replace('= "', 'LIKE "')
        
        # Map human terms to schema
        generated_sql = generated_sql.replace("story", "description")
        generated_sql = generated_sql.replace("plot", "description")
        generated_sql = generated_sql.replace("synopsis", "description")

        # Clean up potential prefixes
        cleaned_sql = generated_sql.replace("SQL:", "").replace("sql:", "").strip()
        return cleaned_sql

if __name__ == "__main__":
    # Test
    gen = SQLGenerator()
    headers = ["title", "mediaType", "eps", "duration", "ongoing", "startYr", "finishYr", "sznOfRelease", "description", "studios", "tags", "contentWarn", "watched", "watching", "wantWatch", "dropped", "rating", "votes"]
    types = ["TEXT", "TEXT", "REAL", "REAL", "INTEGER", "REAL", "REAL", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "REAL", "REAL", "REAL", "REAL", "REAL", "REAL"]
    
    q = "what is the story of one piece"
    print(f"Query: {q}")
    print(f"SQL: {gen.generate_sql(q, 'anime', headers, types)}")
