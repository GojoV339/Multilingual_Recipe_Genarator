"""
Metrics evaluation module for comparing model outputs.
Supports multiple NLP metrics including BLEU, ROUGE, semantic similarity, etc.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# Try to import optional metrics libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. BLEU scores will be simplified.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available. ROUGE metrics will be skipped.")


class ModelEvaluator:
    """Evaluates model outputs using multiple metrics."""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the evaluator with an embedding model for semantic similarity."""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU score between reference and candidate text."""
        if not NLTK_AVAILABLE:
            # Simple token-based overlap as fallback
            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()
            if len(ref_tokens) == 0:
                return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
            
            # Simple unigram precision
            ref_set = set(ref_tokens)
            cand_set = set(cand_tokens)
            overlap = len(ref_set & cand_set)
            precision = overlap / len(cand_set) if len(cand_set) > 0 else 0.0
            
            return {
                "bleu": precision,
                "bleu_1": precision,
                "bleu_2": precision,
                "bleu_3": precision,
                "bleu_4": precision
            }
        
        try:
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            if len(cand_tokens) == 0:
                return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
            
            # Calculate BLEU scores for different n-grams
            bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
            bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing)
            bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing)
            bleu_4 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
            
            return {
                "bleu": bleu_4,
                "bleu_1": bleu_1,
                "bleu_2": bleu_2,
                "bleu_3": bleu_3,
                "bleu_4": bleu_4
            }
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores between reference and candidate text."""
        if not ROUGE_AVAILABLE:
            return {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            }
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                "rouge1": {
                    "precision": scores['rouge1'].precision,
                    "recall": scores['rouge1'].recall,
                    "fmeasure": scores['rouge1'].fmeasure
                },
                "rouge2": {
                    "precision": scores['rouge2'].precision,
                    "recall": scores['rouge2'].recall,
                    "fmeasure": scores['rouge2'].fmeasure
                },
                "rougeL": {
                    "precision": scores['rougeL'].precision,
                    "recall": scores['rougeL'].recall,
                    "fmeasure": scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            }
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate cosine similarity between embeddings of reference and candidate."""
        try:
            embeddings = self.embedding_model.encode([reference, candidate])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_length_metrics(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate length-based metrics."""
        ref_words = len(reference.split())
        cand_words = len(candidate.split())
        ref_chars = len(reference)
        cand_chars = len(candidate)
        
        length_ratio = cand_words / ref_words if ref_words > 0 else 0.0
        char_ratio = cand_chars / ref_chars if ref_chars > 0 else 0.0
        
        return {
            "reference_length_words": ref_words,
            "candidate_length_words": cand_words,
            "reference_length_chars": ref_chars,
            "candidate_length_chars": cand_chars,
            "length_ratio": length_ratio,
            "char_ratio": char_ratio
        }
    
    def evaluate(self, reference: str, candidate: str) -> Dict:
        """Calculate all metrics for a reference-candidate pair."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "bleu_scores": self.calculate_bleu(reference, candidate),
            "rouge_scores": self.calculate_rouge(reference, candidate),
            "semantic_similarity": self.calculate_semantic_similarity(reference, candidate),
            "length_metrics": self.calculate_length_metrics(reference, candidate)
        }
        
        # Calculate overall score (weighted average)
        bleu_score = metrics["bleu_scores"]["bleu"]
        rouge_f1 = metrics["rouge_scores"]["rougeL"]["fmeasure"]
        semantic_sim = metrics["semantic_similarity"]
        
        # Weighted average: 30% BLEU, 30% ROUGE, 40% Semantic Similarity
        overall_score = (0.3 * bleu_score + 0.3 * rouge_f1 + 0.4 * semantic_sim)
        metrics["overall_score"] = overall_score
        
        return metrics


def create_results_folder(base_dir: str = "evaluation_results") -> str:
    """Create a timestamped folder for saving evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"eval_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_evaluation_result(
    folder_path: str,
    model_name: str,
    query: str,
    reference: Optional[str],
    candidate: str,
    metrics: Dict,
    metadata: Optional[Dict] = None
) -> str:
    """Save evaluation results to a JSON file."""
    result = {
        "model_name": model_name,
        "query": query,
        "reference": reference,
        "candidate": candidate,
        "metrics": metrics,
        "metadata": metadata or {}
    }
    
    # Create a safe filename from query
    safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_query = safe_query.replace(' ', '_')
    filename = f"{model_name}_{safe_query}_{datetime.now().strftime('%H%M%S')}.json"
    filepath = os.path.join(folder_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_summary(folder_path: str, all_results: List[Dict]) -> str:
    """Save a summary of all evaluations in a folder."""
    if not all_results:
        return ""
    
    # Calculate aggregate metrics
    bleu_scores = [r["metrics"]["bleu_scores"]["bleu"] for r in all_results]
    rouge_f1_scores = [r["metrics"]["rouge_scores"]["rougeL"]["fmeasure"] for r in all_results]
    semantic_sims = [r["metrics"]["semantic_similarity"] for r in all_results]
    overall_scores = [r["metrics"]["overall_score"] for r in all_results]
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluations": len(all_results),
        "aggregate_metrics": {
            "bleu": {
                "mean": float(np.mean(bleu_scores)),
                "std": float(np.std(bleu_scores)),
                "min": float(np.min(bleu_scores)),
                "max": float(np.max(bleu_scores))
            },
            "rouge_l_f1": {
                "mean": float(np.mean(rouge_f1_scores)),
                "std": float(np.std(rouge_f1_scores)),
                "min": float(np.min(rouge_f1_scores)),
                "max": float(np.max(rouge_f1_scores))
            },
            "semantic_similarity": {
                "mean": float(np.mean(semantic_sims)),
                "std": float(np.std(semantic_sims)),
                "min": float(np.min(semantic_sims)),
                "max": float(np.max(semantic_sims))
            },
            "overall_score": {
                "mean": float(np.mean(overall_scores)),
                "std": float(np.std(overall_scores)),
                "min": float(np.min(overall_scores)),
                "max": float(np.max(overall_scores))
            }
        },
        "model_breakdown": {}
    }
    
    # Group by model
    models = {}
    for result in all_results:
        model = result["model_name"]
        if model not in models:
            models[model] = []
        models[model].append(result)
    
    for model, results in models.items():
        model_bleu = [r["metrics"]["bleu_scores"]["bleu"] for r in results]
        model_rouge = [r["metrics"]["rouge_scores"]["rougeL"]["fmeasure"] for r in results]
        model_sem = [r["metrics"]["semantic_similarity"] for r in results]
        model_overall = [r["metrics"]["overall_score"] for r in results]
        
        summary["model_breakdown"][model] = {
            "count": len(results),
            "mean_bleu": float(np.mean(model_bleu)),
            "mean_rouge_f1": float(np.mean(model_rouge)),
            "mean_semantic_similarity": float(np.mean(model_sem)),
            "mean_overall_score": float(np.mean(model_overall))
        }
    
    summary_path = os.path.join(folder_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary_path

