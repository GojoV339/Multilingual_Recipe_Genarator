import pandas as pd
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import ast
import time
import os



CSV_PATH = 'modified.csv'
DB_PATH = 'recipes.db'
FAISS_INDEX_PATH = 'recipes.index'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 



def create_db(db_path):
    
    if os.path.exists(db_path):
        os.remove(db_path) 
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE recipes (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        ingredients TEXT NOT NULL,
        directions TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database created at {db_path}")

def insert_recipe(db_path, recipe_id, title, ingredients, directions):
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO recipes (id, title, ingredients, directions) VALUES (?, ?, ?, ?)",
        (recipe_id, title, ingredients, directions)
    )
    conn.commit()
    conn.close()



def main():
    print("Starting Offline Recipe Index Build")

    if not os.path.exists(CSV_PATH):
        print(f"Error: Cannot find recipe data at {CSV_PATH}")
        print("Please download the RecipeNLG dataset, save it as 'recipe_nlg.csv', and place it in this directory.")
        return

    
    create_db(DB_PATH)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

    
    print(f"Loading data from {CSV_PATH}...")
    try:
        
        df = pd.read_csv(CSV_PATH)
        
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} recipes. Now creating embeddings...")
    
    embeddings = []
    start_time = time.time()

    
    for index, row in df.iterrows():
        try:

            title = str(row['title'])
            
            
            ner_list = ast.literal_eval(str(row['NER']))
            ner_string = ", ".join(ner_list)
            
            search_text = f"{title}: {ner_string}"
            
            
            embedding = model.encode(search_text, convert_to_tensor=False)
            embeddings.append(embedding)
            
            
            insert_recipe(
                DB_PATH,
                index,
                title,
                str(row['ingredients']), 
                str(row['directions'])  
            )

            if (index + 1) % 1000 == 0:
                print(f"  ...processed {index + 1} recipes.")

        except Exception as e:
            print(f"Warning: Skipping row {index} due to error: {e}")
            print(f"  Row data: {row}")

    end_time = time.time()
    print(f"Embedding creation finished. Took {end_time - start_time:.2f} seconds.")

    
    print("Building FAISS index...")
    
    embeddings_np = np.array(embeddings).astype('float32')
    
    d = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(d)

    index.add(embeddings_np)

    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print("Build Complete! You can now run the Streamlit app")

if __name__ == "__main__":
    main()