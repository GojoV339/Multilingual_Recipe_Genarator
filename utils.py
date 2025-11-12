import os
import time
import streamlit as st
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import ast

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = 'recipes.index'
DB_PATH = 'recipes.db'
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'

def load_api_key():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in .env file.")
        return None
    try:
        genai.configure(api_key=api_key)
        return api_key
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_faiss_index(path):
    if not os.path.exists(path):
        st.error(f"FAISS index not found at {path}.")
        return None
    try:
        index = faiss.read_index(path)
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_resource
def get_db_connection(path):
    if not os.path.exists(path):
        st.error(f"SQLite DB not found at {path}.")
        return None
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"Error connecting to SQLite DB: {e}")
        return None

def search_recipes(query, model, index, k=1):
    if model is None or index is None:
        return []
    try:
        query_emb = model.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = index.search(query_emb, k)
        return indices[0]
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return []

def get_recipe_details(db_conn, recipe_id):
    if db_conn is None:
        return None
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT title, ingredients, directions FROM recipes WHERE id = ?", (int(recipe_id),))
        return cursor.fetchone()
    except Exception as e:
        st.error(f"Error retrieving recipe: {e}")
        return None

def build_rag_prompt(recipe_details, language):
    title, ingredients_str, directions_str = recipe_details

    try:
        ingredients_list = ast.literal_eval(ingredients_str)
    except:
        ingredients_list = [ingredients_str]

    try:
        directions_list = ast.literal_eval(directions_str)
    except:
        directions_list = [directions_str]

    ingredients_fmt = "\n".join([f"- {item}" for item in ingredients_list])
    directions_fmt = "\n".join([f"{i+1}. {step}" for i, step in enumerate(directions_list)])

    prompt = f"""
    **Task:** You are an expert translator. Translate this English recipe into {language}.

    **Title:** {title}

    **Ingredients:**
    {ingredients_fmt}

    **Directions:**
    {directions_fmt}

    **Your {language} Translation:**
    """
    return prompt

async def get_gemini_translation(prompt):
    if not load_api_key():
        return "API Key not configured."
    model = genai.GenerativeModel(GEMINI_MODEL)
    max_retries = 5
    delay = 2

    for _ in range(max_retries):
        try:
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Gemini API error: {e}")
                return str(e)
    return "Service busy. Try again later."

async def get_recipe_name_from_ingredients(ingredients_str, language):
    if not load_api_key():
        return "API Key not configured."
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
    A user has the following ingredients:
    "{ingredients_str}"
    Return only the recipe name they can make.
    If ingredients are in {language}, return a common English recipe name.
    User: "{ingredients_str}"
    You:
    """

    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip().replace('"', '')
    except Exception as e:
        st.error(f"Error getting recipe name: {e}")
        return None

async def get_gemini_response_with_search(query, language):
    if not load_api_key():
        return "API Key not configured."
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
    Find a high-quality recipe for "{query}" using Google Search. Then translate it fully into {language}.
    Return only the translated recipe.
    """

    max_retries = 5
    delay = 2

    for _ in range(max_retries):
        try:
            response = await model.generate_content_async(
                prompt,
                tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())]
            )
            return response.text
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Search error: {e}")
                return str(e)
    return "Service busy. Try again later."
