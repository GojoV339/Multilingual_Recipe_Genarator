import pandas as pd
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import ast
import time
import os

# --- CONFIGURATION ---
CSV_PATH = 'modified.csv' 
DB_PATH = 'recipes.db'
FAISS_INDEX_PATH = 'recipes.index'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'



def create_db_if_not_exists(db_path):
    """Creates the DB and table ONLY if it doesn't already exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if the table 'recipes' exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recipes (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        ingredients TEXT NOT NULL,
        directions TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database {db_path} is ready.")

def get_existing_ids(db_path):
    """Gets all IDs already processed and returns them as a Set for fast lookup."""
    if not os.path.exists(db_path):
        return set() # Empty set
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM recipes")
        ids = cursor.fetchall() # This will be a list of tuples, e.g., [(0,), (1,), ...]
        id_set = {i[0] for i in ids} # Fast set comprehension
        print(f"Found {len(id_set)} existing recipes in the database.")
        return id_set
    except sqlite3.OperationalError:
        # This can happen if the table wasn't created yet
        print("Database file found, but 'recipes' table is missing. Starting fresh.")
        return set()
    finally:
        conn.close()


def insert_recipe(db_path, recipe_id, title, ingredients, directions):
    """Inserts a single recipe into the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO recipes (id, title, ingredients, directions) VALUES (?, ?, ?, ?)",
            (recipe_id, title, ingredients, directions)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # This is a safety check, in case we try to insert a duplicate ID
        print(f"Warning: ID {recipe_id} already exists. Skipping.")
    except Exception as e:
        print(f"Error inserting {recipe_id}: {e}")
    finally:
        conn.close()

# --- Main Processing ---

def main():
    print("--- Starting RESUMABLE Recipe Index Build ---")
    
    # 0. Delete the bad index file (safety check)
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Found old/partial index file at {FAISS_INDEX_PATH}. Deleting it.")
        os.remove(FAISS_INDEX_PATH)

    # 1. Ensure DB and table exist
    create_db_if_not_exists(DB_PATH)

    # 2. Get all IDs we've already processed
    existing_ids = get_existing_ids(DB_PATH)
    
    if 1810000 in existing_ids:
        print("Confirmed: Your progress from last night is loaded.")

    # 3. Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    # 4. Load the full recipe CSV
    print(f"Loading data from {CSV_PATH}...")
    try:
        # Load the entire CSV
        df = pd.read_csv(CSV_PATH)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} recipes. Now processing...")
    print("This will re-generate embeddings (lost in crash) but skip existing DB inserts.")
    
    # This list will be built from scratch (unavoidable)
    embeddings = [] 
    start_time = time.time()
    db_inserts_skipped = 0
    db_inserts_done = 0

    # 5. Loop through data
    for index, row in df.iterrows():
        # 'index' is the row number, which IS our 'id'
        current_id = index
        
        try:
            # --- PART A: Database Check ---
            if current_id not in existing_ids:
                # This row is NOT in the DB. Insert it.
                insert_recipe(
                    DB_PATH,
                    current_id,
                    str(row['title']),
                    str(row['ingredients']),
                    str(row['directions'])
                )
                db_inserts_done += 1
            else:
                # This row IS in the DB. Skip insertion.
                db_inserts_skipped += 1

            # --- PART B: Embedding Generation (Must be done for every row) ---
            title = str(row['title'])
            ner_list = ast.literal_eval(str(row['NER']))
            ner_string = ", ".join(ner_list)
            search_text = f"{title}: {ner_string}"
            
            # This is the slow part that was lost in the crash
            embedding = model.encode(search_text, convert_to_tensor=False)
            embeddings.append(embedding)

            if (current_id + 1) % 50000 == 0:
                print(f"  ...processed {current_id + 1} rows.")
                print(f"     (DB Inserts: {db_inserts_done}, DB Skips: {db_inserts_skipped})")

        except Exception as e:
            print(f"Warning: Skipping row {current_id} due to error: {e}")
            print(f"  Row data: {row}")

    end_time = time.time()
    print(f"Processing finished. Took {end_time - start_time:.2f} seconds.")
    print(f"Total DB Inserts: {db_inserts_done}, Total DB Skips: {db_inserts_skipped}")

    # 6. Create and save the FAISS index
    if not embeddings:
        print("No embeddings were generated. Cannot build FAISS index.")
        return
        
    print("Building final FAISS index...")
    
    embeddings_np = np.array(embeddings).astype('float32')
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print("--- Build Complete! ---")

if __name__ == "__main__":
    main()