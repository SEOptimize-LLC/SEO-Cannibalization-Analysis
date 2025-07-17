# Enhanced SEO Cannibalization Analysis Tool

## 🚀 New Features Added

### 1. URL-Level Consolidation Analysis
The tool now provides **URL-level analysis** instead of just query-level recommendations. This addresses the core issue you mentioned about making consolidation decisions at the URL level.

### 2. Keyword Overlap Percentage
- **Percentage calculation**: Shows the % of keyword overlap between any two URLs
- **Combined metrics**: Sums clicks and impressions for overlapping keywords between URL pairs
- **Jaccard similarity**: Additional similarity metric for comprehensive analysis

### 3. Semantic Similarity Integration
- **Optional embeddings support**: Upload semantic similarity data for enhanced analysis
- **CSV format support**: Compatible with your existing semantic similarity reports
- **Fallback mechanism**: Works without embeddings using basic similarity calculations

### 4. Enhanced Consolidation Recommendations
- **6 action types**: Merge, Redirect, Optimize, Internal Link, Monitor, False Positive
- **Priority levels**: High, Medium, Low based on traffic recovery potential
- **Traffic recovery estimates**: Calculates potential clicks recovery from consolidation

## 📊 How to Use the Enhanced Features

### Basic Usage (No Embeddings)
1. Upload your Google Search Console CSV file
2. Run analysis - you'll get URL-level recommendations with keyword overlap
3. Filter by overlap percentage, action type, or priority

### Advanced Usage (With Embeddings)
1. Upload your GSC CSV file
2. Upload your semantic similarity CSV file (format: Address, Closest Semantically Similar Address, Semantic Similarity Score)
3. Run analysis - enhanced recommendations with semantic similarity scores

## 🔍 Understanding the Output

### URL Consolidation Recommendations Table
- **Primary URL**: The URL to keep (higher performing)
- **Secondary URL**: The URL to consolidate/redirect
- **Keyword Overlap Count**: Number of shared keywords
- **Keyword Overlap %**: Percentage of keyword overlap
- **Semantic Similarity**: Similarity score (when embeddings used)
- **Combined Clicks/Impressions**: Total performance metrics
- **Potential Traffic Recovery**: Estimated clicks recovery
- **Recommended Action**: Specific action to take
- **Priority**: Implementation priority

### Action Types Explained
- **Merge**: Combine content from secondary to primary URL
- **Redirect**: 301 redirect secondary to primary URL
- **Optimize**: Optimize both URLs for different keywords
- **Internal Link**: Add strategic internal linking
- **Monitor**: Keep monitoring performance
- **False Positive**: URLs are distinct, no action needed

## 📁 File Structure

```
SEO-Cannibalization-Analysis/
├── main.py                          # Enhanced main application
├── features/
│   ├── url_consolidation_analyzer.py # New URL-level analysis
│   ├── simple_similarity_loader.py  # Semantic similarity loader
│   └── enhanced_consolidation.py    # Advanced consolidation logic
├── semantic_similarity_template.csv  # Template for embeddings data
└── ENHANCED_README.md               # This file
```

## 🎯 Key Improvements Summary

| Feature | Before | After |
|---------|--------|-------|
| Analysis Level | Query-level only | URL-level with query overlap |
| Consolidation Decision | Manual based on query data | Automated with 6 action types |
| Metrics | Basic clicks/impressions | Keyword overlap % + semantic similarity |
| Traffic Estimation | Not available | Potential recovery calculation |
| Semantic Analysis | Not supported | Optional embeddings integration |

## 🔄 Migration from Original Tool

The enhanced tool is **fully backward compatible** - all original functionality remains intact while adding the new URL-level analysis. Simply:

1. Use the same GSC CSV format
2. Optionally add semantic similarity CSV for enhanced analysis
3. Access URL consolidation results in the new "🔗 URL Consolidation Analysis" tab

## 📈 Example Use Cases

### Scenario 1: High Keyword Overlap
- **URLs**: /blog/seo-tips and /guide/seo-best-practices
- **Overlap**: 85% keyword overlap
- **Recommendation**: Merge content into single comprehensive guide
- **Expected Recovery**: ~350 clicks

### Scenario 2: Moderate Overlap with Embeddings
- **URLs**: /product/shoes and /product/running-shoes
- **Overlap**: 45% keyword overlap
- **Semantic Similarity**: 92%
- **Recommendation**: Redirect /product/shoes to /product/running-shoes
- **Expected Recovery**: ~200 clicks

### Scenario 3: Low Overlap
- **URLs**: /blog/seo and /blog/content-marketing
- **Overlap**: 15% keyword overlap
- **Recommendation**: Monitor and optimize separately
- **Expected Recovery**: ~50 clicks (not worth consolidation)

## 🛠️ Technical Requirements

- Same as original tool (Streamlit, pandas, numpy, plotly)
- Optional: Semantic similarity CSV file for enhanced analysis
- No additional dependencies required

## 🎉 Benefits

1. **Better Decision Making**: URL-level analysis with concrete metrics
2. **Time Savings**: Automated recommendations instead of manual analysis
3. **Traffic Recovery**: Quantified potential gains from consolidation
4. **Flexibility**: Works with or without semantic embeddings
5. **Actionable Insights**: Specific actions with priority levels
