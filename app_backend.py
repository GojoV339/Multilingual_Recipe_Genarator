"""
Flask backend API for Recipe Generator
Provides REST API endpoints for the frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import utils
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  

# Load environment variables
load_dotenv()

_embedding_model = None
_faiss_index = None
_db_connection = None

def initialize_resources():
    """Initialize and cache resources."""
    global _embedding_model, _faiss_index, _db_connection
    
    if _embedding_model is None:
        _embedding_model = utils.get_embedding_model()
    
    if _faiss_index is None:
        _faiss_index = utils.load_faiss_index(utils.FAISS_INDEX_PATH)
    
    if _db_connection is None:
        _db_connection = utils.get_db_connection(utils.DB_PATH)
    
    return _embedding_model, _faiss_index, _db_connection

def run_async(coro):
    """Run async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    return jsonify({
        'providers': {
            'gemini': ['gemini-2.5-flash-preview-09-2025', 'gemini-1.5-pro', 'gemini-1.5-flash'],
            'groq': utils.GROQ_MODELS
        }
    })

@app.route('/api/recipe-name', methods=['POST'])
def get_recipe_name():
    """Get recipe name from ingredients."""
    try:
        data = request.json
        ingredients = data.get('ingredients', '')
        language = data.get('language', 'Telugu')
        provider = data.get('provider', 'gemini')  # 'gemini' or 'groq'
        model_name = data.get('model_name', None)
        
        if not ingredients:
            return jsonify({'error': 'Ingredients are required'}), 400
        
        if provider == 'groq':
            if not model_name:
                model_name = utils.DEFAULT_GROQ_MODEL
            recipe_name = run_async(
                utils.get_recipe_name_from_ingredients_groq(ingredients, language, model_name)
            )
        else:
            recipe_name = run_async(
                utils.get_recipe_name_from_ingredients(ingredients, language)
            )
        
        if not recipe_name:
            return jsonify({'error': 'Could not determine recipe from ingredients'}), 400
        
        return jsonify({'recipe_name': recipe_name})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-internet', methods=['POST'])
def search_internet():
    """Search for recipe on the internet."""
    try:
        data = request.json
        query = data.get('query', '')
        language = data.get('language', 'Telugu')
        enable_eval = data.get('enable_evaluation', False)
        provider = data.get('provider', 'gemini')  # 'gemini' or 'groq'
        model_name = data.get('model_name', None)
        reference_text = data.get('reference_text', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Set default model name if not provided
        if not model_name:
            if provider == 'groq':
                model_name = utils.DEFAULT_GROQ_MODEL
            else:
                model_name = 'gemini-2.5-flash'
        
        # Get recipe from internet based on provider
        if provider == 'groq':
            recipe = run_async(
                utils.get_groq_response_with_search(query, language, model_name)
            )
        else:
            recipe = run_async(
                utils.get_gemini_response_with_search(query, language)
            )
        
        if not recipe or recipe.startswith('Error'):
            return jsonify({'error': recipe or 'Failed to get recipe from internet'}), 500
        
        response_data = {'recipe': recipe}
        
        # Evaluate if enabled
        if enable_eval and reference_text:
            print(f"DEBUG: Evaluation enabled for query: {query}, provider: {provider}, model: {model_name}")
            try:
                eval_result = utils.evaluate_model_output(
                    model_name=f"{provider}:{model_name}",
                    query=query,
                    reference=reference_text,
                    candidate=recipe,
                    language=language,
                    metadata={'source': 'internet', 'provider': provider}
                )
                if eval_result:
                    print(f"DEBUG: Evaluation successful, file saved to: {eval_result.get('filepath', 'unknown')}")
                    response_data['metrics'] = eval_result['metrics']
                else:
                    print("DEBUG: Evaluation returned None (check logs above for errors)")
            except Exception as e:
                print(f"ERROR: Evaluation error: {e}")
                import traceback
                traceback.print_exc()
        elif enable_eval and not reference_text:
            print("DEBUG: Evaluation enabled but no reference text provided")
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-database', methods=['POST'])
def search_database():
    """Search for recipe in database."""
    try:
        data = request.json
        query = data.get('query', '')
        language = data.get('language', 'Telugu')
        enable_eval = data.get('enable_evaluation', False)
        provider = data.get('provider', 'gemini')  # 'gemini' or 'groq'
        model_name = data.get('model_name', None)
        reference_text = data.get('reference_text', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Set default model name if not provided
        if not model_name:
            if provider == 'groq':
                model_name = utils.DEFAULT_GROQ_MODEL
            else:
                model_name = 'gemini-2.5-flash'
        
        # Initialize resources
        model, index, db_conn = initialize_resources()
        
        if not all([model, index, db_conn]):
            return jsonify({'error': 'Database resources not available'}), 500
        
        # Search database
        recipe_ids = utils.search_recipes(query, model, index, k=1)
        
        if not recipe_ids or len(recipe_ids) == 0:
            # Fallback to internet search
            if provider == 'groq':
                recipe = run_async(
                    utils.get_groq_response_with_search(query, language, model_name)
                )
            else:
                recipe = run_async(
                    utils.get_gemini_response_with_search(query, language)
                )
            if not recipe or recipe.startswith('Error'):
                return jsonify({'error': 'Recipe not found in database or internet'}), 404
            
            response_data = {'recipe': recipe, 'source': 'internet_fallback'}
            
            # Evaluate if enabled
            if enable_eval and reference_text:
                print(f"DEBUG: Evaluation enabled for query: {query}, provider: {provider}, model: {model_name}")
                try:
                    eval_result = utils.evaluate_model_output(
                        model_name=f"{provider}:{model_name}",
                        query=query,
                        reference=reference_text,
                        candidate=recipe,
                        language=language,
                        metadata={'source': 'internet_fallback', 'provider': provider}
                    )
                    if eval_result:
                        print(f"DEBUG: Evaluation successful, file saved to: {eval_result.get('filepath', 'unknown')}")
                        response_data['metrics'] = eval_result['metrics']
                    else:
                        print("DEBUG: Evaluation returned None (check logs above for errors)")
                except Exception as e:
                    print(f"ERROR: Evaluation error: {e}")
                    import traceback
                    traceback.print_exc()
            elif enable_eval and not reference_text:
                print("DEBUG: Evaluation enabled but no reference text provided")
            
            return jsonify(response_data)
        
        # Get recipe details from database
        recipe_id = recipe_ids[0]
        recipe_details = utils.get_recipe_details(db_conn, recipe_id)
        
        if not recipe_details:
            return jsonify({'error': 'Recipe details not found'}), 404
        
        # Build prompt and translate based on provider
        prompt = utils.build_rag_prompt(recipe_details, language)
        if provider == 'groq':
            recipe = run_async(utils.get_groq_translation(prompt, model_name))
        else:
            recipe = run_async(utils.get_gemini_translation(prompt))
        
        if not recipe or recipe.startswith('Error'):
            return jsonify({'error': 'Failed to translate recipe'}), 500
        
        # Prepare reference for evaluation
        title, ingredients_str, directions_str = recipe_details
        reference_recipe = f"{title}\n\nIngredients:\n{ingredients_str}\n\nDirections:\n{directions_str}"
        
        response_data = {
            'recipe': recipe,
            'source': 'database',
            'recipe_id': int(recipe_id)
        }
        
        # Evaluate if enabled
        if enable_eval:
            eval_reference = reference_text if reference_text else reference_recipe
            print(f"DEBUG: Evaluation enabled for query: {query}, provider: {provider}, model: {model_name}")
            print(f"DEBUG: Using reference: {'user-provided' if reference_text else 'database recipe'}")
            try:
                eval_result = utils.evaluate_model_output(
                    model_name=f"{provider}:{model_name}",
                    query=query,
                    reference=eval_reference,
                    candidate=recipe,
                    language=language,
                    metadata={'source': 'database', 'recipe_id': int(recipe_id), 'provider': provider}
                )
                if eval_result:
                    print(f"DEBUG: Evaluation successful, file saved to: {eval_result.get('filepath', 'unknown')}")
                    response_data['metrics'] = eval_result['metrics']
                else:
                    print("DEBUG: Evaluation returned None (check logs above for errors)")
            except Exception as e:
                print(f"ERROR: Evaluation error: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize resources on startup
    print("Initializing resources...")
    try:
        initialize_resources()
        print("Resources initialized successfully!")
    except Exception as e:
        print(f"Warning: Some resources failed to load: {e}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

