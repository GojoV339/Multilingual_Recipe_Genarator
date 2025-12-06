# Recipe Generator - Web Application

A beautiful, modern web application for AI-powered recipe generation with 3D effects, animations, and advanced UI.

## Features

- 🎨 **Stunning 3D UI** with glassmorphism effects
- 🌈 **Colorful gradients** and animated backgrounds
- ✨ **Smooth animations** and transitions
- 📊 **Model evaluation** with metrics (BLEU, ROUGE, Semantic Similarity)
- 🌐 **Multi-language support** (Telugu, Hindi)
- 🔍 **Dual search modes**: Recipe name or ingredients
- 💾 **Timestamped evaluation results** saved automatically

## Files Structure

```
dataset/
├── index.html          # Main HTML frontend
├── style.css           # Advanced CSS with 3D effects
├── script.js           # Frontend JavaScript
├── app_backend.py      # Flask backend API
├── utils.py            # Backend utilities
├── metrics.py          # Evaluation metrics
└── requirements.txt    # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install NLTK data (first time only)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configure Environment

Make sure you have a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Start the Backend

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Flask server
python app_backend.py
```

The backend will run on `http://localhost:5000`

### 4. Open the Frontend

Simply open `index.html` in your web browser, or use a local server:

```bash
# Using Python's built-in server
python -m http.server 8000

# Then open: http://localhost:8000/index.html
```

Or use any other static file server.

## Usage

1. **Choose Search Mode**:
   - Recipe Name: Search by dish name
   - My Ingredients: Enter ingredients to find recipes

2. **Enter Query**: Type your search query

3. **Select Language**: Choose Telugu or Hindi for translation

4. **Optional Settings**:
   - Enable "Search Internet" to use web search
   - Enable "Model Evaluation" to get metrics

5. **Click "Get Recipe"**: Wait for the AI to generate your recipe

6. **View Results**: Recipe appears with beautiful formatting and optional metrics

## API Endpoints

The Flask backend provides these endpoints:

- `GET /api/health` - Health check
- `POST /api/recipe-name` - Get recipe name from ingredients
- `POST /api/search-internet` - Search internet for recipes
- `POST /api/search-database` - Search 2.2M recipe database

## Evaluation Metrics

When evaluation is enabled, you get:
- **BLEU Score**: N-gram precision (0-1)
- **ROUGE-L F1**: Longest common subsequence (0-1)
- **Semantic Similarity**: Embedding cosine similarity (0-1)
- **Overall Score**: Weighted combination

Results are automatically saved to `evaluation_results/eval_TIMESTAMP/` folders.

## Design Features

- **3D Logo Animation**: Floating emoji with perspective transform
- **Glassmorphism**: Frosted glass effect on cards
- **Gradient Backgrounds**: Animated floating shapes
- **Smooth Transitions**: All interactions are animated
- **Responsive Design**: Works on desktop and mobile
- **Toast Notifications**: Beautiful notification system
- **Loading Animations**: Engaging loading states

## Troubleshooting

### Backend not starting
- Check if port 5000 is available
- Ensure all dependencies are installed
- Verify `.env` file has correct API key

### CORS errors
- Make sure Flask-CORS is installed
- Check that backend is running on port 5000
- Verify frontend is accessing correct API URL

### Recipe generation fails
- Check Google API key is valid
- Ensure database files exist (`recipes.db`, `recipes.index`)
- Check console for error messages

## Development

To modify the frontend:
- Edit `index.html` for structure
- Edit `style.css` for styling
- Edit `script.js` for functionality

To modify the backend:
- Edit `app_backend.py` for API endpoints
- Edit `utils.py` for core functionality

## Notes

- The Streamlit app (`app.py`) is still available if you prefer that interface
- Both interfaces use the same backend utilities
- Evaluation results are shared between both interfaces

