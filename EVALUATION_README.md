# Model Evaluation System

This evaluation system allows you to compare different models (Gemini, ChatGPT, etc.) using multiple NLP metrics.

## Features

- **Multiple Metrics**: BLEU, ROUGE (1, 2, L), Semantic Similarity, and Length metrics
- **Timestamped Folders**: Each evaluation session creates a unique folder with timestamp
- **No Overwrites**: All results are saved separately, never overwritten
- **Model Comparison**: Easy comparison between different models
- **Summary Reports**: Automatic summary generation with aggregate statistics

## Metrics Explained

### BLEU Score
- Measures n-gram precision between reference and candidate text
- Range: 0.0 to 1.0 (higher is better)
- Includes BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores

### ROUGE Scores
- **ROUGE-1**: Unigram overlap (precision, recall, F1)
- **ROUGE-2**: Bigram overlap (precision, recall, F1)
- **ROUGE-L**: Longest common subsequence (precision, recall, F1)
- Range: 0.0 to 1.0 (higher is better)

### Semantic Similarity
- Cosine similarity between sentence embeddings
- Measures semantic meaning similarity (not just word overlap)
- Range: -1.0 to 1.0 (higher is better, typically 0.0 to 1.0)

### Overall Score
- Weighted average: 30% BLEU + 30% ROUGE-L F1 + 40% Semantic Similarity
- Provides a single metric for quick comparison

## Usage

### In the Streamlit App

1. **Enable Evaluation**: Check "Enable Model Evaluation" in the evaluation settings
2. **Set Model Name**: Enter the model name (e.g., "gemini-2.5-flash", "gpt-4", "claude-3")
3. **Provide Reference**: 
   - Option 1: Enter reference text manually
   - Option 2: Leave empty - will use database recipe as reference (when available)
4. **Run Query**: Get recipe as usual
5. **View Metrics**: Metrics are displayed automatically after evaluation
6. **Check Results**: Results are saved to `evaluation_results/eval_YYYYMMDD_HHMMSS/`

### Folder Structure

```
evaluation_results/
├── eval_20241212_143022/
│   ├── gemini-2.5-flash_Chicken_Biryani_143025.json
│   ├── gemini-2.5-flash_Dosa_143030.json
│   └── summary.json
├── eval_20241212_150000/
│   ├── gpt-4_Chicken_Biryani_150005.json
│   ├── gpt-4_Dosa_150010.json
│   └── summary.json
└── ...
```

### Result File Format

Each result file contains:
```json
{
  "model_name": "gemini-2.5-flash",
  "query": "Chicken Biryani",
  "reference": "Original English recipe text...",
  "candidate": "Model output text...",
  "metrics": {
    "timestamp": "2024-12-12T14:30:25",
    "bleu_scores": {...},
    "rouge_scores": {...},
    "semantic_similarity": 0.85,
    "length_metrics": {...},
    "overall_score": 0.82
  },
  "metadata": {
    "language": "Telugu",
    "source": "database",
    "recipe_id": 12345
  }
}
```

### Summary File

The `summary.json` file contains aggregate statistics:
- Mean, std, min, max for all metrics
- Breakdown by model
- Total number of evaluations

## Comparing Models

1. **Run evaluations with different models** using the same queries
2. **Use different evaluation sessions** (click "New Evaluation Session") for each model
3. **Compare summary.json files** from different folders
4. **Or use the aggregate metrics** in the summary to compare models

## Requirements

Install additional dependencies:
```bash
pip install nltk rouge-score
```

Download NLTK data (first time only):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Tips

- **Consistent Queries**: Use the same queries for different models for fair comparison
- **Reference Quality**: Better reference text = more meaningful metrics
- **Multiple Runs**: Run multiple queries per model for statistical significance
- **Check Summaries**: The summary.json files provide aggregate statistics for easy comparison

## Future Enhancements

- Automatic model comparison dashboard
- Statistical significance testing
- Visualization of metrics over time
- Export to CSV/Excel for analysis

