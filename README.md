# SEO Cannibalization Analysis Tool

A modular Python tool built from scratch for analyzing keyword cannibalization using Google Search Console data and semantic similarity reports.

## 🎯 Features

- **Exact column mapping** as specified:
  - Address → primary_url
  - Closest Semantically Similar Address → secondary_url
  - Semantic Similarity Score → semantic_similarity (2 decimal places)

- **Smart URL filtering** removes:
  - URLs with parameters (=, ?, #, &)
  - Legal pages (privacy, terms, shipping, etc.)
  - About/Contact pages
  - Subdomains
  - Paginated pages (/page/1/, /page/2/)
  - Archive pages (/2025/07/25)
  - Other unwanted patterns

- **Precise action classification** based on exact rules:
  - **Remove**: Both URLs have 0 clicks and 0 indexed keywords
  - **Merge**: 90%+ similarity with at least 1 click and 1 keyword
  - **Redirect**: <90% similarity with at least 1 click and 1 keyword
  - **Internal Link**: ≤89% similarity with significant traffic
  - **Optimize**: ≤89% similarity, low traffic, decent impressions
  - **False Positive**: Default for edge cases

- **Priority assignment** based on traffic potential:
  - **High**: Traffic score ≥ 100
  - **Medium**: Traffic score ≥ 10
  - **Low**: Traffic score < 10

## 📁 Project Structure

```
SEO-Cannibalization-Analysis/
├── main.py                 # Main Streamlit orchestrator
├── requirements.txt        # Dependencies
├── README.md              # This file
└── src/
    ├── data_loader.py      # File loading and validation
    ├── url_filter.py       # URL filtering logic
    ├── metrics_calculator.py  # GSC metrics calculation
    ├── action_classifier.py   # Action classification rules
    ├── priority_assigner.py   # Priority assignment logic
    └── report_generator.py    # Final report generation
```

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run main.py
```

### 3. Upload Files
- **Google Search Console Report**: CSV/Excel with columns: query, page, clicks, impressions
- **Semantic Similarity Report**: CSV/Excel with columns: Address, Closest Semantically Similar Address, Semantic Similarity Score

### 4. Download Results
- Complete analysis in exact format
- Actionable recommendations for each URL pair
- Priority-based sorting

## 📊 Output Format

The final report contains exactly these columns:

| Column | Description |
|--------|-------------|
| primary_url | Primary URL from similarity report |
| primary_url_indexed_queries | Number of queries ranking for primary URL |
| primary_url_clicks | Total clicks for primary URL |
| primary_url_impressions | Total impressions for primary URL |
| secondary_url | Secondary URL from similarity report |
| secondary_url_indexed_queries | Number of queries ranking for secondary URL |
| secondary_url_clicks | Total clicks for secondary URL |
| secondary_url_impressions | Total impressions for secondary URL |
| semantic_similarity | Similarity score (2 decimal places) |
| recommended_action | Action recommendation |
| priority | Priority level (High/Medium/Low) |

## 🎯 Action Rules Summary

| Action | Similarity | Clicks | Keywords | Additional Conditions |
|--------|------------|--------|----------|----------------------|
| Remove | Any | 0 | 0 | Both URLs must have 0 |
| Merge | ≥90% | ≥1 | ≥1 | Both URLs must have ≥1 |
| Redirect | <90% | ≥1 | ≥1 | Both URLs must have ≥1 |
| Internal Link | ≤89% | >1 | >1 | Significant traffic |
| Optimize | ≤89% | ≤1 | ≤1 | Impressions > 100 |
| False Positive | Any | Any | Any | Default fallback |

## 🔧 Technical Details

- **Built with**: Python, Streamlit, Pandas
- **Architecture**: Modular, extensible design
- **Error handling**: Comprehensive validation and user feedback
- **Performance**: Optimized for large datasets
- **Compatibility**: Works with CSV and Excel files

## 📈 Example Usage

1. Export GSC data: Search Results → Export → CSV
2. Run semantic similarity analysis on your URLs
3. Upload both files to the tool
4. Get actionable recommendations for URL consolidation
