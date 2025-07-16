# SEO Cannibalization Analysis Tool

## Overview
This enhanced version of the SEO Cannibalization Analysis Tool provides URL-level consolidation recommendations with advanced metrics including keyword overlap analysis, semantic similarity scoring, and comprehensive traffic impact calculations.

## New Features

### 🔗 URL-Level Consolidation Analysis
The tool now provides detailed URL-level analysis beyond query-level recommendations:

#### Key Metrics:
- **Keyword Overlap Percentage**: Calculates the percentage of overlapping keywords between any two URLs
- **Combined Traffic Metrics**: Shows total clicks and impressions for overlapping keywords
- **Semantic Similarity**: Optional semantic analysis using query content similarity
- **Traffic Recovery Estimates**: Predicts potential traffic recovery from consolidation
- **Confidence Scoring**: Provides confidence levels for each recommendation

#### Recommendation Types:
1. **Merge & Redirect**: High overlap and similarity - implement 301 redirect
2. **Redirect Secondary**: Good overlap - redirect after content audit
3. **Evaluate Content Merge**: Moderate overlap - review before deciding
4. **Monitor & Optimize**: Low overlap - keep separate and optimize individually

### 📊 Enhanced Dashboard
- **Third Tab**: New "URL Consolidation" tab with comprehensive URL-level analysis
- **Interactive Filters**: Filter by priority, minimum recovery potential, and keyword overlap percentage
- **Visual Metrics**: Clear metrics cards showing consolidation opportunities
- **Downloadable Reports**: Export URL-level recommendations as CSV

## How to Use

### 1. Upload Data
- Upload your Google Search Console CSV export
- Configure brand variants to exclude branded queries
- Run the analysis

### 2. Navigate the Tabs
- **Dashboard**: Overview of your data and cleaning results
- **Analysis Results**: Query-level cannibalization analysis
- **URL Consolidation**: Enhanced URL-level recommendations

### 3. URL Consolidation Features
- **Summary Metrics**: View total URL pairs, priority breakdown, and potential recovery
- **Filtering**: Use filters to focus on high-impact opportunities
- **Detailed View**: Expand individual recommendations for detailed insights
- **Export**: Download filtered recommendations as CSV

## Technical Implementation

### EnhancedConsolidationAnalyzer
The new `EnhancedConsolidationAnalyzer` class in `features/enhanced_consolidation.py` provides:

#### Methods:
- `analyze_url_consolidation()`: Main analysis method
- `_calculate_keyword_overlap()`: Computes keyword overlap between URLs
- `_calculate_url_metrics()`: Aggregates performance metrics per URL
- `_generate_enhanced_recommendations()`: Creates consolidation recommendations
- `_calculate_confidence_score()`: Provides confidence ratings

#### Data Structure:
Each URL recommendation includes:
- Primary and secondary URLs
- Keyword overlap count and percentage
- Traffic metrics (clicks, impressions)
- Semantic similarity score
- Consolidation type recommendation
- Priority level
- Confidence score
- Implementation notes

## Example Output

### URL Recommendation Example:
```
Primary URL: /blog/seo-guide
Secondary URL: /seo-tips-2024
Keyword Overlap: 15 keywords (75% overlap)
Primary Clicks: 1,250
Secondary Clicks: 340
Combined Clicks: 1,590
Potential Recovery: 238 clicks
Recommendation: Merge & Redirect
Priority: High
Confidence: 87.5%
```

## Installation & Setup

### Requirements
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Plotly

### Running the Tool
```bash
cd "C:\Users\admin\Documents\Marketing\Roger SEO\Scripts\SEO-Cannibalization-Analysis"
streamlit run main.py
```

## Data Requirements
Your Google Search Console export should contain:
- `query`: Search query
- `page`: Page URL
- `clicks`: Number of clicks
- `impressions`: Number of impressions
- `position`: Average position

## Advanced Features

### Semantic Similarity (Future Enhancement)
The tool is designed to support semantic similarity analysis using embeddings. To enable:
1. Install sentence-transformers: `pip install sentence-transformers`
2. Set `use_semantic_similarity=True` in EnhancedConsolidationAnalyzer

### Custom Thresholds
Adjust these parameters in the code:
- `CLICK_PERCENTAGE_THRESHOLD`: Minimum click share threshold
- `MIN_CLICKS_THRESHOLD`: Minimum clicks for consideration
- Filter thresholds in the UI

## Benefits Over Original Version

| Feature | Original | Enhanced |
|---------|----------|----------|
| Analysis Level | Query-level | URL-level |
| Overlap Metrics | Basic | Comprehensive |
| Traffic Impact | Query-based | URL-based |
| Recommendations | Simple merge/redirect | 4-tier system |
| Filtering | Basic | Advanced |
| Export Options | Single format | Multiple formats |
| Confidence Scoring | None | Weighted scoring |

## Troubleshooting

### Common Issues:
1. **No URL recommendations**: Check minimum overlap and recovery thresholds
2. **Large datasets**: Processing may take longer - consider filtering by date range
3. **Memory issues**: Reduce dataset size or increase system memory

### Performance Tips:
- Filter by date range before export from GSC
- Use brand filtering to reduce dataset size
- Adjust minimum thresholds to focus on high-impact opportunities

## Support
For issues or questions, refer to the original repository or create an issue with detailed information about your dataset and the specific problem encountered.
