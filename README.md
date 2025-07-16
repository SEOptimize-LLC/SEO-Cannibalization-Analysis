# Enhanced SEO Cannibalization Analysis Tool

## Overview
This enhanced version of the SEO Cannibalization Analysis Tool provides URL-level consolidation recommendations with advanced metrics including keyword overlap percentages, semantic similarity scoring, and comprehensive traffic recovery estimates.

## New Features

### 🔗 URL-Level Consolidation Analysis
- **Keyword Overlap Analysis**: Calculate the percentage of keyword overlap between any two URLs
- **Semantic Similarity**: Enhanced similarity scoring using TF-IDF or optional embeddings
- **Traffic Recovery Estimates**: Predict potential traffic recovery from URL consolidation
- **Actionable Recommendations**: Specific actions (Merge, Redirect, Optimize, etc.) with priorities

### 📊 Enhanced Metrics
- **Combined Clicks & Impressions**: Sum traffic metrics for URL pairs
- **Jaccard Similarity**: Measure keyword set similarity between URLs
- **Confidence Scoring**: Weighted scoring system for recommendations
- **Priority Classification**: High/Medium/Low priority based on potential impact

### 🧠 Semantic Analysis Options
- **Basic TF-IDF**: Uses query text for similarity calculation
- **Enhanced Embeddings**: Optional upload of URL embeddings for better semantic similarity
- **Flexible Similarity**: Works with or without external embeddings

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### New Dependencies
- `scikit-learn>=1.3.0` - For TF-IDF vectorization and cosine similarity

## Usage

### Basic Usage
1. Upload your Google Search Console CSV file
2. Run the analysis
3. Navigate to the "URL Consolidation Analysis" tab

### Advanced Usage with Embeddings
1. Upload your GSC CSV file
2. Optionally upload URL embeddings CSV file
3. Run the analysis for enhanced semantic similarity
4. Use the URL consolidation recommendations

### CSV Format for Embeddings
Your embeddings file should have the following format:
```
url,embedding_dim_1,embedding_dim_2,...,embedding_dim_n
https://example.com/page1,0.123,0.456,...,0.789
https://example.com/page2,0.234,0.567,...,0.890
```

## Understanding the Results

### URL Consolidation Recommendations
Each recommendation includes:
- **Primary URL**: The URL to keep (higher traffic)
- **Secondary URL**: The URL to consolidate/redirect
- **Action**: Specific action to take (Merge, Redirect, Optimize, etc.)
- **Priority**: High/Medium/Low based on potential impact
- **Keyword Overlap Count**: Number of shared keywords
- **Keyword Overlap %**: Percentage of keyword overlap
- **Semantic Similarity**: Similarity score (0-100%)
- **Traffic Metrics**: Clicks and impressions for both URLs
- **Potential Recovery**: Estimated traffic recovery from consolidation
- **Shared Keywords**: List of overlapping keywords

### Action Types
- **Merge**: High overlap and similarity - consolidate content
- **Redirect**: Good overlap - redirect secondary to primary
- **Optimize**: Moderate overlap - optimize both URLs separately
- **Internal Link**: Low overlap - add internal linking
- **Monitor**: Minimal overlap - monitor performance
- **False Positive**: Very low overlap - likely not cannibalization

### Priority Levels
- **High**: Potential recovery > 500 clicks
- **Medium**: Potential recovery 100-500 clicks
- **Low**: Potential recovery < 100 clicks

## Example Workflow

### 1. Data Preparation
Upload your Google Search Console export with columns:
- query
- page
- clicks
- impressions
- position

### 2. Optional Embeddings
Create embeddings for your URLs using:
- Sentence transformers (e.g., all-MiniLM-L6-v2)
- Word2Vec on URL content
- Custom embedding models

### 3. Analysis
The tool will:
- Clean and process your data
- Calculate keyword overlaps between URLs
- Generate semantic similarity scores
- Provide actionable consolidation recommendations
- Estimate traffic recovery potential

### 4. Implementation
Use the recommendations to:
- Identify high-impact consolidation opportunities
- Prioritize actions based on potential recovery
- Make data-driven decisions about URL consolidation

## Technical Details

### Algorithm Overview
1. **Data Cleaning**: Remove invalid entries and standardize data
2. **URL Metrics**: Calculate comprehensive metrics for each URL
3. **Keyword Overlap**: Identify shared keywords between URL pairs
4. **Semantic Similarity**: Calculate content similarity using TF-IDF or embeddings
5. **Recommendation Engine**: Generate actionable recommendations with confidence scores
6. **Priority Scoring**: Rank recommendations by potential impact

### Scoring System
- **Keyword Overlap Weight**: 40%
- **Semantic Similarity Weight**: 20%
- **Traffic Recovery Weight**: 40%

## Troubleshooting

### Common Issues
1. **No recommendations found**: Check if you have enough URLs with overlapping keywords
2. **Embeddings not working**: Ensure embeddings file has 'url' column and numeric values
3. **Memory issues**: Reduce dataset size or use basic TF-IDF instead of embeddings

### Performance Tips
- Use basic TF-IDF for large datasets (>10k URLs)
- Upload embeddings for enhanced accuracy on smaller datasets
- Filter by minimum keyword overlap to focus on relevant pairs

## File Structure
```
SEO-Cannibalization-Analysis/
├── main.py                    # Enhanced main application
├── features/
│   ├── url_consolidation_analyzer.py  # URL-level analysis
│   └── enhanced_consolidation.py      # Legacy enhanced module
├── requirements.txt           # Updated dependencies
└── ENHANCED_README.md         # This file
```

## Future Enhancements
- Real-time embedding generation
- Advanced NLP for content analysis
- Historical trend analysis
- Integration with content management systems
- API endpoints for programmatic access
