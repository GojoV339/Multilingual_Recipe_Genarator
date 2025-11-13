// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const queryInput = document.getElementById('queryInput');
const languageSelect = document.getElementById('languageSelect');
const internetSearchBtn = document.getElementById('internetSearchBtn');
let internetSearchEnabled = false;
const searchBtn = document.getElementById('searchBtn');
const modeButtons = document.querySelectorAll('.mode-btn');
const resultsSection = document.getElementById('resultsSection');
const recipeContent = document.getElementById('recipeContent');
const closeResultsBtn = document.getElementById('closeResults');
const loadingOverlay = document.getElementById('loadingOverlay');
const enableEvalCheckbox = document.getElementById('enableEval');
const evalOptions = document.getElementById('evalOptions');
const modelNameInput = document.getElementById('modelName');
const referenceTextInput = document.getElementById('referenceText');
const metricsDisplay = document.getElementById('metricsDisplay');

let currentMode = 'name';

// Advanced UI Enhancements
const addAdvancedInteractions = () => {
    // Parallax effect on scroll
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        const header = document.querySelector('.header');
        if (header) {
            header.style.transform = `translateY(${currentScroll * 0.3}px)`;
            header.style.opacity = 1 - (currentScroll / 300);
        }
        lastScroll = currentScroll;
    });

    // Magnetic button effect
    searchBtn.addEventListener('mousemove', (e) => {
        const rect = searchBtn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        searchBtn.style.transform = `translate(${x * 0.1}px, ${y * 0.1}px)`;
    });

    searchBtn.addEventListener('mouseleave', () => {
        searchBtn.style.transform = '';
    });

    // Input focus animations
    queryInput.addEventListener('focus', () => {
        queryInput.parentElement.style.transform = 'scale(1.02)';
    });

    queryInput.addEventListener('blur', () => {
        queryInput.parentElement.style.transform = 'scale(1)';
    });

    // Ripple effect on buttons
    document.querySelectorAll('button, .mode-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });

    // Smooth number animations for metrics
    const animateValue = (element, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = start + (end - start) * progress;
            element.textContent = value.toFixed(4);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };

    // Store animateValue for use in displayMetrics
    window.animateValue = animateValue;
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updatePlaceholder();
    addAdvancedInteractions();
    
    // Add entrance animations
    const elements = document.querySelectorAll('.main-card, .header');
    elements.forEach((el, index) => {
        el.style.animationDelay = `${index * 0.2}s`;
    });
});

// Event Listeners
function setupEventListeners() {
    // Mode toggle
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            modeButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            updatePlaceholder();
        });
    });

    // Search button
    searchBtn.addEventListener('click', handleSearch);

    // Enter key in input
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Close results
    closeResultsBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
    });

    // Internet search button toggle
    internetSearchBtn.addEventListener('click', () => {
        internetSearchEnabled = !internetSearchEnabled;
        internetSearchBtn.classList.toggle('active', internetSearchEnabled);
    });

    // Evaluation toggle
    enableEvalCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            evalOptions.classList.add('show');
        } else {
            evalOptions.classList.remove('show');
        }
    });
}

function updatePlaceholder() {
    if (currentMode === 'name') {
        queryInput.placeholder = 'e.g., Chicken Biryani or Dosa';
    } else {
        queryInput.placeholder = 'e.g., chicken, rice, tomatoes, onions';
    }
}

// Search Handler
async function handleSearch() {
    const query = queryInput.value.trim();
    
    if (!query) {
        showToast('Please enter a search query', 'warning');
        return;
    }

    const searchMode = currentMode;
    const language = languageSelect.value;
    const useInternet = internetSearchEnabled;
    const enableEval = enableEvalCheckbox.checked;
    const modelName = modelNameInput.value.trim() || 'gemini-2.5-flash';
    const referenceText = referenceTextInput.value.trim();

    // Show loading
    showLoading(true);
    searchBtn.classList.add('loading');

    try {
        let response;
        
        if (searchMode === 'ingredients') {
            // First get recipe name from ingredients
            const recipeName = await getRecipeNameFromIngredients(query, language);
            if (!recipeName) {
                throw new Error('Could not determine recipe from ingredients');
            }
            showToast(`Searching for: ${recipeName}`, 'success');
            
            if (useInternet) {
                response = await searchInternet(recipeName, language, enableEval, modelName, referenceText);
            } else {
                response = await searchDatabase(recipeName, language, enableEval, modelName, referenceText);
            }
        } else {
            if (useInternet) {
                response = await searchInternet(query, language, enableEval, modelName, referenceText);
            } else {
                response = await searchDatabase(query, language, enableEval, modelName, referenceText);
            }
        }

        displayRecipe(response.recipe, response.metrics);
        showToast('Recipe generated successfully!', 'success');
        
    } catch (error) {
        console.error('Search error:', error);
        showToast(error.message || 'An error occurred. Please try again.', 'error');
    } finally {
        showLoading(false);
        searchBtn.classList.remove('loading');
    }
}

// API Calls
async function getRecipeNameFromIngredients(ingredients, language) {
    const response = await fetch(`${API_BASE_URL}/recipe-name`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            ingredients,
            language
        })
    });

    if (!response.ok) {
        throw new Error('Failed to get recipe name');
    }

    const data = await response.json();
    return data.recipe_name;
}

async function searchInternet(query, language, enableEval, modelName, referenceText) {
    const response = await fetch(`${API_BASE_URL}/search-internet`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            language,
            enable_evaluation: enableEval,
            model_name: modelName,
            reference_text: referenceText
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to search internet');
    }

    return await response.json();
}

async function searchDatabase(query, language, enableEval, modelName, referenceText) {
    const response = await fetch(`${API_BASE_URL}/search-database`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            language,
            enable_evaluation: enableEval,
            model_name: modelName,
            reference_text: referenceText
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to search database');
    }

    return await response.json();
}

// Display Functions
function displayRecipe(recipeText, metrics = null) {
    // Format and display recipe text
    recipeContent.innerHTML = formatRecipeText(recipeText);
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Display metrics if available
    if (metrics) {
        displayMetrics(metrics);
    } else {
        metricsDisplay.style.display = 'none';
    }
}

function displayMetrics(metrics) {
    metricsDisplay.style.display = 'block';
    
    // Animate metric values
    const bleuEl = document.getElementById('bleuScore');
    const rougeEl = document.getElementById('rougeScore');
    const semanticEl = document.getElementById('semanticScore');
    const overallEl = document.getElementById('overallScore');
    
    const bleu = metrics.bleu_scores?.bleu || 0;
    const rouge = metrics.rouge_scores?.rougeL?.fmeasure || 0;
    const semantic = metrics.semantic_similarity || 0;
    const overall = metrics.overall_score || 0;
    
    if (window.animateValue) {
        window.animateValue(bleuEl, 0, bleu, 1000);
        window.animateValue(rougeEl, 0, rouge, 1000);
        window.animateValue(semanticEl, 0, semantic, 1000);
        window.animateValue(overallEl, 0, overall, 1000);
    } else {
        bleuEl.textContent = bleu.toFixed(4);
        rougeEl.textContent = rouge.toFixed(4);
        semanticEl.textContent = semantic.toFixed(4);
        overallEl.textContent = overall.toFixed(4);
    }
}

// UI Helpers
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    const container = document.getElementById('toastContainer');
    container.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

// Format recipe text with markdown-like formatting
function formatRecipeText(text) {
    // Convert markdown-style formatting to HTML
    let formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    return `<p>${formatted}</p>`;
}

