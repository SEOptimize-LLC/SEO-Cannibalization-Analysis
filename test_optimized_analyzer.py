#!/usr/bin/env python3
"""
Test script for the optimized consolidation analyzer
This script demonstrates how to use the optimized analyzer with semantic similarity data
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.optimized_consolidation_analyzer import OptimizedConsolidationAnalyzer

def test_with_sample_data():
    """Test the optimized analyzer with sample data"""
    
    # Load sample GSC data with URLs that match the semantic similarity data
    print("Loading sample GSC data...")
    gsc_data = pd.DataFrame({
        'query': [
            'does insurance cover invisalign',
            'invisalign vs braces cost',
            'invisalign review',
            'rooibos tea stain teeth',
            'tea stain teeth',
            'herbal tea stain teeth'
        ],
        'page': [
            'https://www.trysnow.com/blogs/news/does-insurance-cover-invisalign',
            'https://www.trysnow.com/blogs/news/invisalign-vs-braces-cost',
            'https://www.trysnow.com/blogs/news/invisalign-review',
            'https://www.trysnow.com/blogs/news/does-rooibos-tea-stain-teeth',
            'https://www.trysnow.com/blogs/news/does-tea-stain-your-teeth',
            'https://www.trysnow.com/blogs/news/does-herbal-tea-stain-teeth'
        ],
        'clicks': [150, 120, 200, 180, 160, 140],
        'impressions': [5000, 4000, 6000, 5500, 5200, 4800],
        'position': [3.2, 2.8, 2.5, 2.1, 2.3, 2.7]
    })
    
    # Load semantic similarity data
    print("Loading semantic similarity data...")
    try:
        similarity_data = pd.read_csv('semantically_similar_report.csv')
        print(f"Loaded {len(similarity_data)} similarity pairs")
        
        # Clean the similarity score (convert comma to dot for decimal)
        similarity_data['Semantic Similarity Score'] = similarity_data['Semantic Similarity Score'].astype(str).str.replace(',', '.').astype(float)
        
        # Rename columns to match expected format
        similarity_data = similarity_data.rename(columns={
            'Address': 'primary_url',
            'Closest Semantically Similar Address': 'secondary_url',
            'Semantic Similarity Score': 'semantic_similarity'
        })
        
        # Add keyword overlap percentage (estimated from semantic similarity)
        similarity_data['keyword_overlap_percentage'] = similarity_data['semantic_similarity'] * 100
        
        # Show first few rows
        print("\nFirst 5 similarity pairs:")
        print(similarity_data[['primary_url', 'secondary_url', 'semantic_similarity']].head())
        
    except Exception as e:
        print(f"Error loading semantic similarity file: {e}")
        similarity_data = pd.DataFrame()  # Empty DataFrame instead of None
    
    # Initialize analyzer
    analyzer = OptimizedConsolidationAnalyzer()
    
    # Run analysis
    print("\nRunning consolidation analysis...")
    results = analyzer.analyze_consolidation(gsc_data, similarity_data)
    
    # Display results
    print(f"\nAnalysis complete!")
    print(f"Total URL pairs analyzed: {results['summary']['total_pairs']}")
    print(f"Recommendations generated: {len(results['recommendations'])}")
    
    if len(results['recommendations']) > 0:
        print("\nTop recommendations:")
        print(results['recommendations'][['primary_url', 'secondary_url', 'semantic_similarity', 'recommended_action', 'priority']].head())
    
    return results

if __name__ == "__main__":
    test_with_sample_data()
