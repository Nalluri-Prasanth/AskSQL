import pandas as pd
import random
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def generate_synthetic_data(csv_path, output_path, num_samples=1000):
    df = pd.read_csv(csv_path)
    
    # Headers matches database.py logic
    # title, mediaType, eps, duration, ongoing, startYr, finishYr, sznOfRelease, description, studios, tags, contentWarn, watched, watching, wantWatch, dropped, rating, votes
    
    examples = []
    
    # Get lists of unique values for templates
    media_types = df['mediaType'].unique().tolist()
    # clean tags
    all_tags = set()
    for tags in df['tags'].dropna():
        # tags are like "['Action', 'Comedy']"
        try:
            t_list = eval(tags)
            all_tags.update(t_list)
        except:
            pass
    tag_list = list(all_tags)
    
    titles = df['title'].dropna().tolist()
    
    # Templates
    # (Question, SQL)
    

    # Helper to randomize case
    def random_case(s):
        if not s: return s
        if random.random() < 0.5:
            return s.lower()
        if random.random() < 0.2:
            return s.upper()
        return s

    # 1. Simple Selects
    examples.append(("Show me all anime", "SELECT * FROM anime"))
    examples.append(("List all anime titles", "SELECT title FROM anime"))
    examples.append(("What are the ratings of all anime?", "SELECT rating FROM anime"))
    
    # 2. Filtering by specific column (Use LIKE for case insensitivity)
    sampled_media_types = random.sample(media_types, min(len(media_types), 5))
    for mt in sampled_media_types:
        if not isinstance(mt, str): continue
        # Training data: Input has random case, SQL has LIKE 'OriginalValue' (or input value, but LIKE makes it flexible)
        # Actually better to teach model to output LIKE 'InputSubstring'.
        # But here we want to overfit slightly to known values if possible, or just generalize.
        # Let's use the value from the INPUT in the SQL if we can, but T5 generation is fuzzy.
        # Simplest approach: Target uses the exact CSV value for 'correctness' in training, but with LIKE.
        # AND we lowercase/randomcase the input.
        
        q_val = random_case(mt)
        examples.append((f"List all {q_val} anime", f"SELECT * FROM anime WHERE mediaType LIKE '{mt}'"))
        examples.append((f"Show me anime that are {q_val}", f"SELECT title FROM anime WHERE mediaType LIKE '{mt}'"))
        examples.append((f"anime aired in {q_val}", f"SELECT title FROM anime WHERE mediaType LIKE '{mt}'"))
        examples.append((f"anime aired on {q_val}", f"SELECT title FROM anime WHERE mediaType LIKE '{mt}'"))
        examples.append((f"Count {q_val} anime", f"SELECT count(*) FROM anime WHERE mediaType LIKE '{mt}'"))

    # 3. Tags (LIKE is already used here, ensuring mixed case inputs)
    for tag in tag_list[:5]: # Expanded to 20 for better variance
        q_tag = random_case(tag)
        examples.append((f"Show me {q_tag} anime", f"SELECT title FROM anime WHERE tags LIKE '%{tag}%'"))
        examples.append((f"List {q_tag} movies", f"SELECT title FROM anime WHERE tags LIKE '%{tag}%' AND mediaType LIKE 'Movie'"))
        examples.append((f"Best {q_tag} anime", f"SELECT title FROM anime WHERE tags LIKE '%{tag}%' ORDER BY rating DESC LIMIT 5"))

    # 4. Ordering (Boosted for robustness)
    examples.append(("List top 5 rated anime", "SELECT title, rating FROM anime ORDER BY rating DESC LIMIT 5"))
    examples.append(("Show me the top 5 anime", "SELECT title, rating FROM anime ORDER BY rating DESC LIMIT 5"))
    examples.append(("rank top 5 anime by rating", "SELECT title, rating FROM anime ORDER BY rating DESC LIMIT 5"))
    examples.append(("best 5 anime", "SELECT title, rating FROM anime ORDER BY rating DESC LIMIT 5"))
    examples.append(("What are the longest anime by episodes?", "SELECT title, eps FROM anime ORDER BY eps DESC LIMIT 5"))
    examples.append(("Show recent anime", "SELECT title, startYr FROM anime ORDER BY startYr DESC LIMIT 10"))

    # 5. Specific Titles
    # Sample some titles
    sampled_titles = random.sample(titles, min(len(titles), 5))
    for t in sampled_titles:
        clean_t = t.replace("'", "''")
        q_t = random_case(t)
        # Use LIKE for titles too to catch partials or casing
        examples.append((f"What is the rating of {q_t}?", f"SELECT rating FROM anime WHERE title LIKE '{clean_t}'"))
        examples.append((f"What is the story of {q_t}?", f"SELECT description FROM anime WHERE title LIKE '{clean_t}'"))
        examples.append((f"Tell me the plot of {q_t}", f"SELECT description FROM anime WHERE title LIKE '{clean_t}'"))
        examples.append((f"Give me the synopsis of {q_t}", f"SELECT description FROM anime WHERE title LIKE '{clean_t}'"))
        examples.append((f"How many episodes does {q_t} have?", f"SELECT eps FROM anime WHERE title LIKE '{clean_t}'"))
        examples.append((f"Who made {q_t}?", f"SELECT studios FROM anime WHERE title LIKE '{clean_t}'"))

    # 6. Combinations
    examples.append(("Top 3 TV shows", "SELECT title FROM anime WHERE mediaType LIKE 'TV' ORDER BY rating DESC LIMIT 3"))
    examples.append(("Most popular anime", "SELECT title, votes FROM anime ORDER BY votes DESC LIMIT 5"))
    examples.append(("Best anime from 2020", "SELECT title FROM anime WHERE startYr = 2020 ORDER BY rating DESC LIMIT 5"))
    
    # Create Table Schema for T5-awesome-text-to-sql
    # Format: tables: CREATE TABLE table_name (col1 type1, col2 type2, ...) query for: question
    
    col_defs = []
    for col, dtype in df.dtypes.items():
        col_clean = col.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        if "float" in str(dtype) or "int" in str(dtype):
            t = "REAL" 
        else:
            t = "TEXT"
        col_defs.append(f"{col_clean} {t}")
    
    create_table_str = f"CREATE TABLE anime ({', '.join(col_defs)})"
    
    processed_data = []
    for q, sql in examples:
        input_text = f"tables: {create_table_str} query for: {q}"
        processed_data.append({
            'input_text': input_text,
            'target_text': sql
        })
        
    out_df = pd.DataFrame(processed_data)
    out_df.to_csv(output_path, index=False)
    print(f"Generated {len(out_df)} synthetic examples to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data('data/animenew.csv', 'data/train_processed.csv', num_samples=40)
    # Use same for dev for now as we just want to overfit/learn this specific table
    generate_synthetic_data('data/animenew.csv', 'data/dev_processed.csv', num_samples=10)
