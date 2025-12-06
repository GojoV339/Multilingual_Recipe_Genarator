# Quick Start Guide - Recipe Generator Web App

## 🚀 How to Run (3 Simple Steps)

### Step 1: Install Dependencies
```bash
# Activate your virtual environment
source .venv/bin/activate

# Install Flask (if not already installed)
pip install flask flask-cors
```

### Step 2: Start the Backend Server
```bash
# Make sure you're in the dataset directory
cd /home/toji339/Documents/Sem-5/Text\ Analytics/Project/Dataset/dataset

# Activate virtual environment
source .venv/bin/activate

# Start Flask backend
python app_backend.py
```

You should see:
```
Initializing resources...
Resources initialized successfully!
 * Running on http://0.0.0.0:5000
```

**Keep this terminal open!** The backend must be running.

### Step 3: Open the Frontend

**Option A: Direct File Open (Easiest)**
- Simply double-click `index.html` or right-click → "Open with" → your browser
- Or drag and drop `index.html` into your browser

**Option B: Using Python HTTP Server**
Open a NEW terminal window:
```bash
cd /home/toji339/Documents/Sem-5/Text\ Analytics/Project/Dataset/dataset
python -m http.server 8000
```
Then open: `http://localhost:8000/index.html` in your browser

## ✅ That's It!

The app should now be running:
- **Backend**: Running on `http://localhost:5000`
- **Frontend**: Open `index.html` in your browser

## 🎯 Using the App

1. **Choose search mode**: Recipe Name or My Ingredients
2. **Enter your query**: e.g., "Chicken Biryani"
3. **Select language**: Telugu or Hindi
4. **Optional**: Enable "Search Internet" or "Model Evaluation"
5. **Click "Get Recipe"** and wait for the magic! ✨

## 🔧 Troubleshooting

### Backend won't start?
- Check if port 5000 is already in use: `lsof -i :5000`
- Make sure `.env` file exists with `GOOGLE_API_KEY=your_key`
- Ensure `recipes.db` and `recipes.index` files exist

### Frontend can't connect to backend?
- Make sure backend is running on port 5000
- Check browser console (F12) for errors
- Verify API URL in `script.js` is `http://localhost:5000/api`

### CORS errors?
- Make sure `flask-cors` is installed
- Check that backend is running

## 📝 Notes

- The backend must be running for the frontend to work
- You can keep both Streamlit app and web app - they use the same backend
- Evaluation results are saved to `evaluation_results/` folder

