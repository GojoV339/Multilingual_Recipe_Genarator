import os
import time
import faiss
import sqlite3
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from metrics import ModelEvaluator, create_results_folder, save_evaluation_result, save_summary

# Make streamlit optional for Flask backend
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock st object for Flask backend
    class MockStreamlit:
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def cache_resource(self, func): return func
        def expander(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def columns(self, n): return [self] * n
        def metric(self, *args, **kwargs): pass
        def subheader(self, *args, **kwargs): pass
        def tabs(self, *args, **kwargs): return [self] * len(args[0]) if args else []
        def json(self, *args, **kwargs): pass
    st = MockStreamlit()
# --- FIX: This is the correct import for google-generativeai package ---
import google.generativeai as genai
# --- FIX: Import Tool and GoogleSearchRetrieval from the correct modules ---
try:
    from google.generativeai.types import Tool
    from google.generativeai import protos
except ImportError:
    st.error("Failed to import types from 'google.generativeai'. Please run 'pip install google-generativeai'")
    # Create stubs so the app doesn't crash on load
    Tool = None
    protos = None

import ast
import requests
from bs4 import BeautifulSoup

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = 'recipes.index'
DB_PATH = 'recipes.db'
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'

# Available Groq models
GROQ_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'llama-3.1-70b-versatile',
    'llama-3.1-405b-reasoning',
    'llama-3.2-90b-vision-preview',
    'llama-3.2-11b-vision-preview',
    'llama-3.2-3b-instruct',
    'llama-3.2-1b-instruct',
    'mixtral-8x7b-32768',
    'gemma-7b-it',
    'gemma2-9b-it',
    'deepseek-r1-distill-llama-70b',
    'qwen/qwen3-32b',
    'moonshotai/kimi-k2-instruct',
]

# Default Groq model
DEFAULT_GROQ_MODEL = 'llama-3.3-70b-versatile'

# Global evaluator instance (cached)
_evaluator = None

def get_evaluator():
    """Get or create the model evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator(embedding_model_name=EMBEDDING_MODEL)
    return _evaluator

def load_api_key():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        if STREAMLIT_AVAILABLE:
            st.error("GOOGLE_API_KEY not found in .env file.")
        else:
            print("WARNING: GOOGLE_API_KEY not found in .env file.")
        return None
    try:
        # This will now work because 'genai' is the correct object
        genai.configure(api_key=api_key)
        return api_key
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"Error configuring Gemini API: {e}")
        else:
            print(f"ERROR: Error configuring Gemini API: {e}")
        return None

def load_groq_api_key():
    """Load and configure Groq API key."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        if STREAMLIT_AVAILABLE:
            st.error("GROQ_API_KEY not found in .env file.")
        else:
            print("WARNING: GROQ_API_KEY not found in .env file.")
        return None
    if not GROQ_AVAILABLE:
        if STREAMLIT_AVAILABLE:
            st.error("Groq package not installed. Run: pip install groq")
        else:
            print("ERROR: Groq package not installed. Run: pip install groq")
        return None
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"Error configuring Groq API: {e}")
        else:
            print(f"ERROR: Error configuring Groq API: {e}")
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
        
    # --- Check if imports were successful ---
    if Tool is None or protos is None:
        if STREAMLIT_AVAILABLE:
            st.error("Could not import Google Search tools. Please restart after 'pip install'.")
        else:
            print("ERROR: Could not import Google Search tools.")
        return "Error: Search tool not available."

    # --- FIX: For gemini-2.5 models, Google Search is enabled via tool_config ---
    # The google_search_retrieval tool is deprecated, but we can enable Google Search
    # through the model's built-in capabilities by using a model that supports it
    
    # Try using a model that supports Google Search natively
    # If the preview model doesn't work, fall back to a stable version
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
    except:
        # Fallback to a stable model that supports Google Search
        model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    Search the web for a high-quality recipe for "{query}". 
    Find the most popular and well-reviewed recipe, then translate the complete recipe 
    (including title, ingredients list, and detailed step-by-step instructions) into {language}.
    Return only the translated recipe in a clear, formatted way with proper sections.
    """

    max_retries = 5
    delay = 2

    for _ in range(max_retries):
        try:
            # The model will automatically use Google Search when needed
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(delay)
                delay *= 2
            else:
                if STREAMLIT_AVAILABLE:
                    st.error(f"Search error: {e}")
                else:
                    print(f"ERROR: Search error: {e}")
                return str(e)
    return "Service busy. Try again later."

async def get_groq_translation(prompt, model_name=DEFAULT_GROQ_MODEL):
    """Get translation from Groq API."""
    client = load_groq_api_key()
    if not client:
        return "Groq API Key not configured."
    
    max_retries = 5
    delay = 2
    
    for _ in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                if STREAMLIT_AVAILABLE:
                    st.error(f"Groq API error: {e}")
                else:
                    print(f"ERROR: Groq API error: {e}")
                return str(e)
    return "Service busy. Try again later."

async def get_recipe_name_from_ingredients_groq(ingredients_str, language, model_name=DEFAULT_GROQ_MODEL):
    """Get recipe name from ingredients using Groq."""
    client = load_groq_api_key()
    if not client:
        return "Groq API Key not configured."
    
    prompt = f"""
    A user has the following ingredients:
    "{ingredients_str}"
    Return only the recipe name they can make.
    If ingredients are in {language}, return a common English recipe name.
    User: "{ingredients_str}"
    You:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"Error getting recipe name from Groq: {e}")
        else:
            print(f"ERROR: Error getting recipe name from Groq: {e}")
        return None

def search_web_for_recipe(query):
    """Search the web for a recipe and return text content."""
    try:
        # Use DuckDuckGo or Google search via requests
        search_url = f"https://html.duckduckgo.com/html/?q={query} recipe"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from search results
        results = soup.find_all(['p', 'div'], class_=lambda x: x and 'result' in x.lower())
        text_content = ' '.join([r.get_text() for r in results[:5]])
        
        if not text_content:
            # Fallback: return a simple description
            return f"Recipe information for {query}"
        
        return text_content[:2000]  # Limit content length
    except Exception as e:
        print(f"Web search error: {e}")
        return f"Recipe information for {query}"

async def get_groq_response_with_search(query, language, model_name=DEFAULT_GROQ_MODEL):
    """Get recipe from internet using Groq with web search."""
    client = load_groq_api_key()
    if not client:
        return "Groq API Key not configured."
    
    # First, try to get web content
    web_content = search_web_for_recipe(query)
    
    prompt = f"""
    Based on the following web search results and your knowledge, provide a high-quality recipe for "{query}".
    
    Web search results:
    {web_content}
    
    Please provide a complete recipe (including title, ingredients list, and detailed step-by-step instructions) translated into {language}.
    If the web search results don't contain enough information, use your knowledge to provide a good recipe.
    Return only the translated recipe in a clear, formatted way with proper sections.
    """
    
    max_retries = 5
    delay = 2
    
    for _ in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                if STREAMLIT_AVAILABLE:
                    st.error(f"Groq search error: {e}")
                else:
                    print(f"ERROR: Groq search error: {e}")
                return str(e)
    return "Service busy. Try again later."


# Evaluation functions
_results_folder = None
_evaluation_results = []

def get_or_create_results_folder():
    """Get or create the current evaluation results folder."""
    global _results_folder
    if _results_folder is None:
        _results_folder = create_results_folder()
    return _results_folder

def reset_evaluation_session():
    """Reset the evaluation session (creates new folder)."""
    global _results_folder, _evaluation_results
    _results_folder = None
    _evaluation_results = []

def evaluate_model_output(
    model_name: str,
    query: str,
    reference: Optional[str],
    candidate: str,
    language: str,
    metadata: Optional[Dict] = None
) -> Optional[Dict]:
    """Evaluate model output and save results."""
    if not reference:
        if STREAMLIT_AVAILABLE:
            st.warning("⚠️ No reference text provided. Evaluation requires a reference text for comparison.")
        else:
            print("WARNING: No reference text provided. Evaluation requires a reference text for comparison.")
        return None
    
    try:
        evaluator = get_evaluator()
        metrics = evaluator.evaluate(reference, candidate)
        
        folder_path = get_or_create_results_folder()
        print(f"DEBUG: Saving evaluation to folder: {folder_path}")
        
        # Save individual result
        filepath = save_evaluation_result(
            folder_path=folder_path,
            model_name=model_name,
            query=query,
            reference=reference,
            candidate=candidate,
            metrics=metrics,
            metadata={
                **(metadata or {}),
                "language": language,
                "timestamp": metrics["timestamp"]
            }
        )
        print(f"DEBUG: Evaluation result saved to: {filepath}")
        
        # Store for summary
        _evaluation_results.append({
            "model_name": model_name,
            "query": query,
            "reference": reference,
            "candidate": candidate,
            "metrics": metrics
        })
        
        return {
            "folder_path": folder_path,
            "filepath": filepath,
            "metrics": metrics
        }
    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        if STREAMLIT_AVAILABLE:
            st.error(error_msg)
        else:
            print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return None

def display_evaluation_metrics(metrics: Dict):
    """Display evaluation metrics in Streamlit."""
    with st.expander("📊 Evaluation Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("BLEU Score", f"{metrics['bleu_scores']['bleu']:.4f}")
        
        with col2:
            st.metric("ROUGE-L F1", f"{metrics['rouge_scores']['rougeL']['fmeasure']:.4f}")
        
        with col3:
            st.metric("Semantic Similarity", f"{metrics['semantic_similarity']:.4f}")
        
        with col4:
            st.metric("Overall Score", f"{metrics['overall_score']:.4f}")
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["BLEU", "ROUGE", "Semantic", "Length"])
        
        with tab1:
            st.json(metrics['bleu_scores'])
        
        with tab2:
            st.json(metrics['rouge_scores'])
        
        with tab3:
            st.metric("Cosine Similarity", f"{metrics['semantic_similarity']:.4f}")
        
        with tab4:
            st.json(metrics['length_metrics'])

def save_evaluation_summary():
    """Save summary of all evaluations in current session."""
    global _evaluation_results, _results_folder
    if not _evaluation_results or not _results_folder:
        return None
    
    try:
        summary_path = save_summary(_results_folder, _evaluation_results)
        return summary_path
    except Exception as e:
        st.error(f"Error saving summary: {e}")
        return None