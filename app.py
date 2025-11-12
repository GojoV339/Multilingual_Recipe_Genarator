import streamlit as st
import utils
import asyncio

async def main():
    st.set_page_config(
        page_title="Recipe Generator",
        page_icon="🍳",
        layout="centered"
    )

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

    if st.button("Get Recipe", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter something to search for.")
        else:
            recipe_name_to_search = query
            
            try:
                if search_mode == "My Ingredients":
                    with st.spinner(f"Asking the chef what can be made with your ingredients..."):
                        recipe_name_to_search = await utils.get_recipe_name_from_ingredients(query, language)
                        if not recipe_name_to_search:
                            st.error("Sorry, I couldn't come up with a recipe for those ingredients.")
                            return
                        st.info(f"Based on your ingredients, I'm searching for: **{recipe_name_to_search}**")
                
                if use_internet_search:
                    st.markdown(f"Searching the internet for **{recipe_name_to_search}**...")
                    with st.spinner(f"Searching Google and translating to {language}..."):
                        response = await utils.get_gemini_response_with_search(recipe_name_to_search, language)
                        st.success("Here's your recipe from the web!")
                        st.markdown(response)

                else:
                    st.markdown(f"Searching our 2.2M recipe database for **{recipe_name_to_search}**...")
                    with st.spinner(f"Searching database and translating to {language}..."):
                        recipe_ids = utils.search_recipes(recipe_name_to_search, model, index, k=1)
                        
                        if not recipe_ids:
                            st.warning("I couldn't find that in our database. Searching the internet instead...")
                            response = await utils.get_gemini_response_with_search(recipe_name_to_search, language)
                            st.success("Here's your recipe from the web!")
                            st.markdown(response)
                        else:
                            recipe_id = recipe_ids[0]
                            recipe_details = utils.get_recipe_details(db_conn, recipe_id)

                            if not recipe_details:
                                st.error("Found a match but couldn't retrieve its details. The database might be out of sync.")
                                return
                            
                            prompt = utils.build_rag_prompt(recipe_details, language)
                            response = await utils.get_gemini_translation(prompt)
                            
                            st.success("Here's your recipe from our database!")
                            st.markdown(response)

                            with st.expander("Show Original English Recipe (from database)"):
                                title, ingredients_str, directions_str = recipe_details
                                st.subheader(title)
                                st.markdown("**Ingredients:**")
                                st.json(ingredients_str)
                                st.markdown("**Directions:**")
                                st.json(directions_str)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)

if __name__ == "__main__":
    asyncio.run(main())
