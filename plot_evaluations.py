"""
Plot Evaluation Results
Reads evaluation JSON files and creates comprehensive visualizations
"""

import os
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_evaluation_data(base_dir="evaluation_results"):
    """Load all evaluation JSON files from all folders."""
    data = []
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found!")
        return []
    
    # Find all JSON files in evaluation_results folders
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    # Filter out summary.json files
    json_files = [f for f in json_files if not f.endswith("summary.json")]
    
    print(f"Found {len(json_files)} evaluation JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
                # Extract key metrics
                metrics = result.get('metrics', {})
                metadata = result.get('metadata', {})
                
                data.append({
                    'file': os.path.basename(json_file),
                    'folder': os.path.basename(os.path.dirname(json_file)),
                    'model_name': result.get('model_name', 'unknown'),
                    'provider': metadata.get('provider', 'unknown'),
                    'query': result.get('query', 'unknown'),
                    'language': metadata.get('language', 'unknown'),
                    'source': metadata.get('source', 'unknown'),
                    'bleu': metrics.get('bleu_scores', {}).get('bleu', 0),
                    'bleu_1': metrics.get('bleu_scores', {}).get('bleu_1', 0),
                    'bleu_2': metrics.get('bleu_scores', {}).get('bleu_2', 0),
                    'bleu_3': metrics.get('bleu_scores', {}).get('bleu_3', 0),
                    'bleu_4': metrics.get('bleu_scores', {}).get('bleu_4', 0),
                    'rouge1_f1': metrics.get('rouge_scores', {}).get('rouge1', {}).get('fmeasure', 0),
                    'rouge2_f1': metrics.get('rouge_scores', {}).get('rouge2', {}).get('fmeasure', 0),
                    'rougeL_f1': metrics.get('rouge_scores', {}).get('rougeL', {}).get('fmeasure', 0),
                    'semantic_similarity': metrics.get('semantic_similarity', 0),
                    'overall_score': metrics.get('overall_score', 0),
                    'reference_length': metrics.get('length_metrics', {}).get('reference_length_words', 0),
                    'candidate_length': metrics.get('length_metrics', {}).get('candidate_length_words', 0),
                    'length_ratio': metrics.get('length_metrics', {}).get('length_ratio', 0),
                    'timestamp': metrics.get('timestamp', '')
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return pd.DataFrame(data)

def plot_model_comparison(df, output_dir="evaluation_plots"):
    """Create comparison plots for different models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("No data to plot!")
        return
    
    # Clean model names for better display
    df['model_display'] = df['model_name'].str.replace('groq:', 'Groq: ').str.replace('gemini:', 'Gemini: ')
    df['model_display'] = df['model_display'].str.replace('_', ' ').str.title()
    
    # 1. Overall Score Comparison by Model
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    model_scores = df.groupby('model_display')['overall_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    model_scores['mean'].plot(kind='barh', yerr=model_scores['std'], capsize=5, color='steelblue')
    plt.xlabel('Overall Score')
    plt.title('Overall Score by Model (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 2. BLEU Score Comparison
    plt.subplot(2, 3, 2)
    bleu_data = df.groupby('model_display')['bleu'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bleu_data['mean'].plot(kind='barh', yerr=bleu_data['std'], capsize=5, color='coral')
    plt.xlabel('BLEU Score')
    plt.title('BLEU Score by Model (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 3. ROUGE-L F1 Comparison
    plt.subplot(2, 3, 3)
    rouge_data = df.groupby('model_display')['rougeL_f1'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    rouge_data['mean'].plot(kind='barh', yerr=rouge_data['std'], capsize=5, color='mediumseagreen')
    plt.xlabel('ROUGE-L F1 Score')
    plt.title('ROUGE-L F1 by Model (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 4. Semantic Similarity Comparison
    plt.subplot(2, 3, 4)
    sem_data = df.groupby('model_display')['semantic_similarity'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    sem_data['mean'].plot(kind='barh', yerr=sem_data['std'], capsize=5, color='gold')
    plt.xlabel('Semantic Similarity')
    plt.title('Semantic Similarity by Model (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 5. Provider Comparison (Gemini vs Groq)
    plt.subplot(2, 3, 5)
    provider_scores = df.groupby('provider')['overall_score'].agg(['mean', 'std'])
    provider_scores['mean'].plot(kind='bar', yerr=provider_scores['std'], capsize=5, color=['#4285F4', '#00A67E'])
    plt.ylabel('Overall Score')
    plt.title('Overall Score by Provider (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # 6. Metric Distribution Box Plot
    plt.subplot(2, 3, 6)
    metrics_to_plot = ['bleu', 'rougeL_f1', 'semantic_similarity', 'overall_score']
    plot_data = df[['model_display'] + metrics_to_plot].melt(
        id_vars='model_display',
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Score'
    )
    sns.boxplot(data=plot_data, x='Metric', y='Score', hue='model_display')
    plt.title('Metric Distribution by Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    plt.suptitle('Model Evaluation Comparison Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_dashboard.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'model_comparison_dashboard.png')}")
    plt.close()

def plot_detailed_metrics(df, output_dir="evaluation_plots"):
    """Create detailed metric breakdowns."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        return
    
    df['model_display'] = df['model_name'].str.replace('groq:', 'Groq: ').str.replace('gemini:', 'Gemini: ')
    df['model_display'] = df['model_display'].str.replace('_', ' ').str.title()
    
    # 1. BLEU Scores Breakdown
    plt.figure(figsize=(14, 8))
    bleu_cols = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
    bleu_df = df.groupby('model_display')[bleu_cols].mean()
    bleu_df.plot(kind='bar', width=0.8)
    plt.ylabel('BLEU Score')
    plt.xlabel('Model')
    plt.title('BLEU Score Breakdown (1-4 grams) by Model', fontsize=14, fontweight='bold')
    plt.legend(title='BLEU-n', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bleu_breakdown.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'bleu_breakdown.png')}")
    plt.close()
    
    # 2. ROUGE Scores Comparison
    plt.figure(figsize=(14, 8))
    rouge_cols = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    rouge_df = df.groupby('model_display')[rouge_cols].mean()
    rouge_df.plot(kind='bar', width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.ylabel('ROUGE F1 Score')
    plt.xlabel('Model')
    plt.title('ROUGE Scores (1, 2, L) by Model', fontsize=14, fontweight='bold')
    plt.legend(title='ROUGE Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rouge_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'rouge_comparison.png')}")
    plt.close()
    
    # 3. Heatmap of All Metrics
    plt.figure(figsize=(12, 8))
    metrics_for_heatmap = ['bleu', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'semantic_similarity', 'overall_score']
    heatmap_data = df.groupby('model_display')[metrics_for_heatmap].mean()
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Metrics Heatmap by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Metric')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'metrics_heatmap.png')}")
    plt.close()

def plot_query_analysis(df, output_dir="evaluation_plots"):
    """Analyze performance by query/language."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        return
    
    df['model_display'] = df['model_name'].str.replace('groq:', 'Groq: ').str.replace('gemini:', 'Gemini: ')
    df['model_display'] = df['model_display'].str.replace('_', ' ').str.title()
    
    # Performance by Query
    plt.figure(figsize=(14, 8))
    query_scores = df.groupby('query')['overall_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    query_scores['mean'].plot(kind='barh', yerr=query_scores['std'], capsize=5, color='steelblue')
    plt.xlabel('Overall Score')
    plt.title('Overall Score by Query (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'query_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'query_analysis.png')}")
    plt.close()
    
    # Performance by Language
    if 'language' in df.columns and df['language'].nunique() > 1:
        plt.figure(figsize=(12, 8))
        lang_scores = df.groupby('language')['overall_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        lang_scores['mean'].plot(kind='bar', yerr=lang_scores['std'], capsize=5, color='coral')
        plt.ylabel('Overall Score')
        plt.xlabel('Language')
        plt.title('Overall Score by Language (Mean ± Std)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'language_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {os.path.join(output_dir, 'language_analysis.png')}")
        plt.close()

def plot_by_language(df, output_dir="evaluation_plots"):
    """Create separate plots for each language."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty or 'language' not in df.columns:
        print("No language data available for language-specific plots")
        return
    
    df['model_display'] = df['model_name'].str.replace('groq:', 'Groq: ').str.replace('gemini:', 'Gemini: ')
    df['model_display'] = df['model_display'].str.replace('_', ' ').str.title()
    
    languages = df['language'].unique()
    print(f"\nGenerating language-specific plots for {len(languages)} languages...")
    
    for language in languages:
        lang_df = df[df['language'] == language].copy()
        
        if lang_df.empty:
            continue
        
        lang_name_clean = language.replace(' ', '_').replace('/', '_')
        lang_output_dir = os.path.join(output_dir, f'language_{lang_name_clean}')
        os.makedirs(lang_output_dir, exist_ok=True)
        
        print(f"  Processing {language} ({len(lang_df)} evaluations)...")
        
        # 1. Model Comparison Dashboard for this Language
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        model_scores = lang_df.groupby('model_display')['overall_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        if not model_scores.empty:
            model_scores['mean'].plot(kind='barh', yerr=model_scores['std'], capsize=5, color='steelblue')
            plt.xlabel('Overall Score')
            plt.title('Overall Score by Model', fontsize=11, fontweight='bold')
        
        plt.subplot(2, 3, 2)
        bleu_data = lang_df.groupby('model_display')['bleu'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        if not bleu_data.empty:
            bleu_data['mean'].plot(kind='barh', yerr=bleu_data['std'], capsize=5, color='coral')
            plt.xlabel('BLEU Score')
            plt.title('BLEU Score by Model', fontsize=11, fontweight='bold')
        
        plt.subplot(2, 3, 3)
        rouge_data = lang_df.groupby('model_display')['rougeL_f1'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        if not rouge_data.empty:
            rouge_data['mean'].plot(kind='barh', yerr=rouge_data['std'], capsize=5, color='mediumseagreen')
            plt.xlabel('ROUGE-L F1 Score')
            plt.title('ROUGE-L F1 by Model', fontsize=11, fontweight='bold')
        
        plt.subplot(2, 3, 4)
        sem_data = lang_df.groupby('model_display')['semantic_similarity'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        if not sem_data.empty:
            sem_data['mean'].plot(kind='barh', yerr=sem_data['std'], capsize=5, color='gold')
            plt.xlabel('Semantic Similarity')
            plt.title('Semantic Similarity by Model', fontsize=11, fontweight='bold')
        
        plt.subplot(2, 3, 5)
        provider_scores = lang_df.groupby('provider')['overall_score'].agg(['mean', 'std'])
        if not provider_scores.empty:
            provider_scores['mean'].plot(kind='bar', yerr=provider_scores['std'], capsize=5, color=['#4285F4', '#00A67E'])
            plt.ylabel('Overall Score')
            plt.title('Overall Score by Provider', fontsize=11, fontweight='bold')
            plt.xticks(rotation=0)
        
        plt.subplot(2, 3, 6)
        metrics_to_plot = ['bleu', 'rougeL_f1', 'semantic_similarity', 'overall_score']
        plot_data = lang_df[['model_display'] + metrics_to_plot].melt(
            id_vars='model_display',
            value_vars=metrics_to_plot,
            var_name='Metric',
            value_name='Score'
        )
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='Metric', y='Score', hue='model_display')
            plt.title('Metric Distribution by Model', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle(f'Model Evaluation Dashboard - {language}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_dashboard.png'), dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_dashboard.png")
        plt.close()
        
        # 2. Detailed BLEU Breakdown for this Language
        plt.figure(figsize=(14, 8))
        bleu_cols = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
        bleu_df = lang_df.groupby('model_display')[bleu_cols].mean()
        if not bleu_df.empty:
            bleu_df.plot(kind='bar', width=0.8)
            plt.ylabel('BLEU Score')
            plt.xlabel('Model')
            plt.title(f'BLEU Score Breakdown (1-4 grams) by Model - {language}', fontsize=14, fontweight='bold')
            plt.legend(title='BLEU-n', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_bleu_breakdown.png'), dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_bleu_breakdown.png")
        plt.close()
        
        # 3. ROUGE Scores for this Language
        plt.figure(figsize=(14, 8))
        rouge_cols = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
        rouge_df = lang_df.groupby('model_display')[rouge_cols].mean()
        if not rouge_df.empty:
            rouge_df.plot(kind='bar', width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            plt.ylabel('ROUGE F1 Score')
            plt.xlabel('Model')
            plt.title(f'ROUGE Scores (1, 2, L) by Model - {language}', fontsize=14, fontweight='bold')
            plt.legend(title='ROUGE Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_rouge_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_rouge_comparison.png")
        plt.close()
        
        # 4. Heatmap for this Language
        plt.figure(figsize=(12, 8))
        metrics_for_heatmap = ['bleu', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'semantic_similarity', 'overall_score']
        heatmap_data = lang_df.groupby('model_display')[metrics_for_heatmap].mean()
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
            plt.title(f'Metrics Heatmap by Model - {language}', fontsize=14, fontweight='bold')
            plt.ylabel('Metric')
            plt.xlabel('Model')
            plt.tight_layout()
            plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_heatmap.png'), dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_heatmap.png")
        plt.close()
        
        # 5. Query Analysis for this Language
        if lang_df['query'].nunique() > 1:
            plt.figure(figsize=(14, 8))
            query_scores = lang_df.groupby('query')['overall_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
            if not query_scores.empty:
                query_scores['mean'].plot(kind='barh', yerr=query_scores['std'], capsize=5, color='steelblue')
                plt.xlabel('Overall Score')
                plt.title(f'Overall Score by Query - {language}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_query_analysis.png'), dpi=300, bbox_inches='tight')
                print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_query_analysis.png")
            plt.close()
        
        # 6. Model Comparison Bar Chart for this Language
        plt.figure(figsize=(14, 8))
        comparison_metrics = ['bleu', 'rougeL_f1', 'semantic_similarity', 'overall_score']
        comparison_data = lang_df.groupby('model_display')[comparison_metrics].mean()
        if not comparison_data.empty:
            comparison_data.plot(kind='bar', width=0.8)
            plt.ylabel('Score')
            plt.xlabel('Model')
            plt.title(f'All Metrics Comparison by Model - {language}', fontsize=14, fontweight='bold')
            plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(lang_output_dir, f'{lang_name_clean}_metrics_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved: {lang_output_dir}/{lang_name_clean}_metrics_comparison.png")
        plt.close()

def plot_correlation_analysis(df, output_dir="evaluation_plots"):
    """Analyze correlations between metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        return
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    metrics_for_corr = ['bleu', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'semantic_similarity', 'overall_score']
    corr_matrix = df[metrics_for_corr].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'correlation_matrix.png')}")
    plt.close()

def generate_summary_report(df, output_dir="evaluation_plots"):
    """Generate a text summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        return
    
    report_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Evaluations: {len(df)}\n")
        f.write(f"Unique Models: {df['model_name'].nunique()}\n")
        f.write(f"Unique Queries: {df['query'].nunique()}\n")
        f.write(f"Unique Languages: {df['language'].nunique()}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MODEL PERFORMANCE RANKING (by Overall Score)\n")
        f.write("-" * 80 + "\n")
        model_ranking = df.groupby('model_name').agg({
            'overall_score': ['mean', 'std', 'count'],
            'bleu': 'mean',
            'rougeL_f1': 'mean',
            'semantic_similarity': 'mean'
        }).sort_values(('overall_score', 'mean'), ascending=False)
        
        for idx, (model, row) in enumerate(model_ranking.iterrows(), 1):
            f.write(f"\n{idx}. {model}\n")
            f.write(f"   Overall Score: {row[('overall_score', 'mean')]:.4f} ± {row[('overall_score', 'std')]:.4f}\n")
            f.write(f"   BLEU: {row[('bleu', 'mean')]:.4f}\n")
            f.write(f"   ROUGE-L F1: {row[('rougeL_f1', 'mean')]:.4f}\n")
            f.write(f"   Semantic Similarity: {row[('semantic_similarity', 'mean')]:.4f}\n")
            f.write(f"   Evaluations: {int(row[('overall_score', 'count')])}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("PROVIDER COMPARISON\n")
        f.write("-" * 80 + "\n")
        provider_stats = df.groupby('provider')['overall_score'].agg(['mean', 'std', 'count'])
        for provider, row in provider_stats.iterrows():
            f.write(f"\n{provider.upper()}:\n")
            f.write(f"   Mean Overall Score: {row['mean']:.4f} ± {row['std']:.4f}\n")
            f.write(f"   Total Evaluations: {int(row['count'])}\n")
    
    print(f"✓ Saved: {report_path}")

def main():
    """Main function to generate all plots."""
    print("=" * 80)
    print("EVALUATION RESULTS PLOTTING TOOL")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading evaluation data...")
    df = load_evaluation_data()
    
    if df.empty:
        print("No evaluation data found!")
        return
    
    print(f"Loaded {len(df)} evaluation records")
    print(f"Models: {', '.join(df['model_name'].unique())}")
    print()
    
    # Create output directory
    output_dir = "evaluation_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    print("Generating plots...")
    print()
    
    plot_model_comparison(df, output_dir)
    plot_detailed_metrics(df, output_dir)
    plot_query_analysis(df, output_dir)
    plot_correlation_analysis(df, output_dir)
    plot_by_language(df, output_dir)  # New: Language-specific plots
    generate_summary_report(df, output_dir)
    
    print()
    print("=" * 80)
    print(f"✓ All plots saved to '{output_dir}' directory!")
    print("=" * 80)

if __name__ == "__main__":
    main()

