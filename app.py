import streamlit as st
import utils
import asyncio
import html

def run_async(coro):
    """Helper function to run async functions in Streamlit context"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we can't use asyncio.run()
            # Create a new event loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)

def display_recipe_text(text):
    """Display recipe text with reduced font size."""
    # Wrap in a div with class for CSS targeting
    # Escape HTML but preserve newlines for markdown
    st.markdown(f'<div class="recipe-text-small">', unsafe_allow_html=True)
    st.markdown(text)  # Streamlit processes markdown
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Recipe Generator",
        page_icon="🍳",
        layout="centered"
    )
    
    # Add custom CSS to reduce font size for recipe text
    st.markdown("""
    <style>
    /* Target only markdown containers that are recipe text (after success messages) */
    /* Use a more specific selector to avoid affecting other markdown */
    .element-container:has(> div[data-testid="stSuccess"]) + .element-container div[data-testid="stMarkdownContainer"],
    .element-container:has(> div[data-testid="stSuccess"]) ~ .element-container div[data-testid="stMarkdownContainer"] {
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    .element-container:has(> div[data-testid="stSuccess"]) + .element-container div[data-testid="stMarkdownContainer"] p,
    .element-container:has(> div[data-testid="stSuccess"]) ~ .element-container div[data-testid="stMarkdownContainer"] p {
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }
    .element-container:has(> div[data-testid="stSuccess"]) + .element-container div[data-testid="stMarkdownContainer"] h1,
    .element-container:has(> div[data-testid="stSuccess"]) + .element-container div[data-testid="stMarkdownContainer"] h2,
    .element-container:has(> div[data-testid="stSuccess"]) + .element-container div[data-testid="stMarkdownContainer"] h3,
    .element-container:has(> div[data-testid="stSuccess"]) ~ .element-container div[data-testid="stMarkdownContainer"] h1,
    .element-container:has(> div[data-testid="stSuccess"]) ~ .element-container div[data-testid="stMarkdownContainer"] h2,
    .element-container:has(> div[data-testid="stSuccess"]) ~ .element-container div[data-testid="stMarkdownContainer"] h3 {
        font-size: 18px !important;
        margin-top: 12px !important;
        margin-bottom: 8px !important;
    }
    /* Fallback: use a class-based approach */
    .recipe-text-small {
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    .recipe-text-small p, .recipe-text-small div, .recipe-text-small span {
        font-size: 14px !important;
    }
    .recipe-text-small h1, .recipe-text-small h2, .recipe-text-small h3 {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🍳 Recipe Generator")
    st.markdown("Find recipes from our 2.2M database, or search the web for new ideas! Translated to Telugu or Hindi.")

    with st.spinner("Warming up the kitchen... (Loading models...)"):
        api_key = utils.load_api_key()
        model = utils.get_embedding_model()
        index = utils.load_faiss_index(utils.FAISS_INDEX_PATH)
        db_conn = utils.get_db_connection(utils.DB_PATH)

    if not all([api_key, model, index, db_conn]):
        st.error("A critical resource failed to load. The app cannot continue. Please check the console for errors.")
        st.stop()
    
    search_mode = st.radio(
        "How do you want to search?",
        ("Recipe Name", "My Ingredients"),
        horizontal=True,
        key="search_mode"
    )

    if search_mode == "Recipe Name":
        placeholder = "e.g., Chicken Biryani or Dosa"
    else:
        placeholder = "e.g., chicken, rice, tomatoes, onions"
        
    query = st.text_input(
        "What are you looking for?",
        placeholder=placeholder,
        key="query_input"
    )

    col1, col2 = st.columns([1, 2])
    
    with col1:
        language = st.selectbox(
            "Language:",
            ("Telugu", "Hindi"),
            key="language_select"
        )
    
    with col2:
        use_internet_search = st.checkbox(
            "Search internet for new recipes?",
            value=False,
            help="If checked, this will use Google Search to find the most popular recipe, which may be slower. If unchecked, it uses the 2.2M recipe database."
        )
    
    # Evaluation settings
    with st.expander("📊 Model Evaluation Settings", expanded=False):
        enable_evaluation = st.checkbox(
            "Enable Model Evaluation",
            value=False,
            help="Evaluate model output using BLEU, ROUGE, and semantic similarity metrics. Results will be saved to timestamped folders."
        )
        
        if enable_evaluation:
            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                model_name = st.text_input(
                    "Model Name",
                    value="gemini-2.5-flash",
                    help="Name of the model being evaluated (e.g., gemini-2.5-flash, gpt-4, etc.)"
                )
            with col_eval2:
                if st.button("🔄 New Evaluation Session", help="Start a new evaluation session (creates new timestamped folder)"):
                    utils.reset_evaluation_session()
                    st.success("New evaluation session started!")
            
            reference_text = st.text_area(
                "Reference Text (Optional)",
                help="Original English recipe text to compare against. If left empty, will use the database recipe as reference when available.",
                height=100
            )
            
            st.info("💡 **Tip:** Each evaluation creates a timestamped folder. Use 'New Evaluation Session' to start a fresh folder for comparing different models.")

    if st.button("Get Recipe", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter something to search for.")
        else:
            recipe_name_to_search = query
            reference_recipe = None  # Will store original recipe for evaluation
            evaluation_results = None  # Will store evaluation metrics
            
            try:
                if search_mode == "My Ingredients":
                    with st.spinner(f"Asking the chef what can be made with your ingredients..."):
                        recipe_name_to_search = run_async(utils.get_recipe_name_from_ingredients(query, language))
                        if not recipe_name_to_search:
                            st.error("Sorry, I couldn't come up with a recipe for those ingredients.")
                            return
                        st.info(f"Based on your ingredients, I'm searching for: **{recipe_name_to_search}**")
                
                if use_internet_search:
                    st.markdown(f"Searching the internet for **{recipe_name_to_search}**...")
                    with st.spinner(f"Searching Google and translating to {language}..."):
                        response = run_async(utils.get_gemini_response_with_search(recipe_name_to_search, language))
                        st.success("Here's your recipe from the web!")
                        display_recipe_text(response)
                        
                        # Evaluate if enabled
                        if enable_evaluation:
                            with st.spinner("Evaluating model output..."):
                                evaluation_results = utils.evaluate_model_output(
                                    model_name=model_name,
                                    query=recipe_name_to_search,
                                    reference=reference_text if reference_text else None,
                                    candidate=response,
                                    language=language
                                )
                                if evaluation_results:
                                    st.success(f"✅ Evaluation complete! Results saved to: `{evaluation_results['folder_path']}`")
                                    utils.display_evaluation_metrics(evaluation_results['metrics'])

                else:
                    st.markdown(f"Searching our 2.2M recipe database for **{recipe_name_to_search}**...")
                    with st.spinner(f"Searching database and translating to {language}..."):
                        recipe_ids = utils.search_recipes(recipe_name_to_search, model, index, k=1)
                        
                        if not recipe_ids:
                            st.warning("I couldn't find that in our database. Searching the internet instead...")
                            response = run_async(utils.get_gemini_response_with_search(recipe_name_to_search, language))
                            st.success("Here's your recipe from the web!")
                            display_recipe_text(response)
                            
                            # Evaluate if enabled
                            if enable_evaluation:
                                with st.spinner("Evaluating model output..."):
                                    evaluation_results = utils.evaluate_model_output(
                                        model_name=model_name,
                                        query=recipe_name_to_search,
                                        reference=reference_text if reference_text else None,
                                        candidate=response,
                                        language=language
                                    )
                                    if evaluation_results:
                                        st.success(f"✅ Evaluation complete! Results saved to: `{evaluation_results['folder_path']}`")
                                        utils.display_evaluation_metrics(evaluation_results['metrics'])
                        else:
                            recipe_id = recipe_ids[0]
                            recipe_details = utils.get_recipe_details(db_conn, recipe_id)

                            if not recipe_details:
                                st.error("Found a match but couldn't retrieve its details. The database might be out of sync.")
                                return
                            
                            prompt = utils.build_rag_prompt(recipe_details, language)
                            response = run_async(utils.get_gemini_translation(prompt))
                            
                            # Store reference recipe for evaluation
                            title, ingredients_str, directions_str = recipe_details
                            reference_recipe = f"{title}\n\nIngredients:\n{ingredients_str}\n\nDirections:\n{directions_str}"
                            
                            st.success("Here's your recipe from our database!")
                            display_recipe_text(response)

                            with st.expander("Show Original English Recipe (from database)"):
                                st.subheader(title)
                                st.markdown("**Ingredients:**")
                                st.json(ingredients_str)
                                st.markdown("**Directions:**")
                                st.json(directions_str)
                            
                            # Evaluate if enabled
                            if enable_evaluation:
                                with st.spinner("Evaluating model output..."):
                                    # Use provided reference or database recipe
                                    eval_reference = reference_text if reference_text else reference_recipe
                                    evaluation_results = utils.evaluate_model_output(
                                        model_name=model_name,
                                        query=recipe_name_to_search,
                                        reference=eval_reference,
                                        candidate=response,
                                        language=language,
                                        metadata={"source": "database", "recipe_id": int(recipe_id)}
                                    )
                                    if evaluation_results:
                                        st.success(f"✅ Evaluation complete! Results saved to: `{evaluation_results['folder_path']}`")
                                        utils.display_evaluation_metrics(evaluation_results['metrics'])

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
