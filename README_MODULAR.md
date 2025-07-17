# SEO Cannibalization Analysis Tool - Modular Version

A comprehensive Python-based tool for analyzing keyword cannibalization issues using Google Search Console data and semantic similarity scores.

## Overview

This modular version provides a flexible, command-line interface for analyzing SEO cannibalization issues. It processes Google Search Console (GSC) data alongside semantic similarity scores to identify URL pairs that may be competing for the same keywords and provides actionable recommendations.

## Architecture

The tool is built with a modular architecture:

```
src/
├── data_loaders/          # File loading and validation
├── processors/            # Data filtering and cleaning
├── analyzers/             # Action classification and priority assignment
├── formatters/            # Report formatting and export
└── utils/                 # Shared utilities and column mapping
```

## Features

### 🔍 **Flexible Data Loading**
- Support for CSV and Excel files
- Automatic column name normalization
- Robust error handling and validation

### 🧹 **Smart URL Filtering**
- Excludes technical URLs (parameters, subdomains, etc.)
- Filters out legal and navigation pages
- Removes low-quality or spam URLs

### 🎯 **Intelligent Action Classification**
- **Merge**: High similarity (≥90%) with significant traffic
- **Redirect**: Lower similarity with competing content
- **Internal Link**: Related content that should be connected
- **Optimize**: Low-performing content with potential
- **Remove**: Zero-value content
- **False Positive**: Similar but not competing content

### 📊 **Priority-Based Analysis**
- **High**: >20% of total traffic
- **Medium**: 1-20% of total traffic
- **Low**: <1% of total traffic

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis.git
cd SEO-Cannibalization-Analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --gsc gsc_data.csv --similarity similarity_data.csv
```

### Advanced Usage
```bash
python main.py \
  --gsc gsc_data.xlsx \
  --similarity similarity_data.xlsx \
  --output custom_report.csv \
  --gsc-type xlsx \
  --similarity-type xlsx
```

### Command Line Arguments
- `--gsc`: Path to Google Search Console file (required)
- `--similarity`: Path to semantic similarity file (required)
- `--output`: Output file path (default: cannibalization_report.csv)
- `--gsc-type`: GSC file type: csv or xlsx (default: csv)
- `--similarity-type`: Similarity file type: csv or xlsx (default: csv)

## Input Data Requirements

### Google Search Console Data
Required columns (case-insensitive):
- `query` or `keyword`: Search query
- `page` or `url`: Landing page URL
- `clicks`: Number of clicks
- `impressions`: Number of impressions
- `position`: Average position (optional)

### Semantic Similarity Data
Required columns (case-insensitive):
- `primary_url` or `url1`: First URL in pair
- `secondary_url` or `url2`: Second URL in pair
- `similarity_score`: Similarity score (0-1 or 0-100)

## Output Format

The tool generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `primary_url` | The main URL (higher traffic) |
| `primary_url_indexed_queries` | Number of keywords this URL ranks for |
| `primary_url_clicks` | Total clicks for this URL |
| `primary_url_impressions` | Total impressions for this URL |
| `secondary_url` | The competing URL |
| `secondary_url_indexed_queries` | Number of keywords this URL ranks for |
| `secondary_url_clicks` | Total clicks for this URL |
| `secondary_url_impressions` | Total impressions for this URL |
| `semantic_similarity` | Similarity score between URLs (0-1) |
| `recommended_action` | Suggested action (Merge/Redirect/Internal Link/etc.) |
| `priority` | Priority level (High/Medium/Low) |

## Example Workflow

### 1. Prepare Your Data
Export Google Search Console data:
- Go to Google Search Console
- Navigate to Performance > Search Results
- Set date range (recommended: 3-6 months)
- Export as CSV

### 2. Generate Semantic Similarity Data
Use your preferred tool to generate semantic similarity scores between URLs. The output should be a CSV with URL pairs and similarity scores.

### 3. Run Analysis
```bash
python main.py --gsc gsc_export.csv --similarity similarity_scores.csv
```

### 4. Review Results
The output will include:
- Summary statistics
- Detailed recommendations for each URL pair
- Priority-based action items

## Sample Output

```
Starting SEO Cannibalization Analysis...
Loaded 15,234 rows from GSC data
Loaded 1,847 URL pairs from similarity data
Filtered to 12,891 valid URLs
Aggregated metrics for 892 unique URLs
Validated 1,234 URL pairs
Merged data for 1,234 URL pairs
Classifying actions and assigning priorities...
Formatting final report...
Exporting results...

==================================================
ANALYSIS COMPLETE
==================================================
Total URL pairs analyzed: 1,234

Recommended actions:
   Merge: 45
   Redirect: 123
   Internal Link: 234
   Optimize: 89
   Remove: 12
   False Positive: 731

Priority distribution:
   High: 89
   Medium: 345
   Low: 800
==================================================
Report saved to: cannibalization_report.csv
```

## Development

### Adding New Filters
Create new processors in `src/processors/` following the URLFilter pattern.

### Custom Action Classification
Modify `src/analyzers/action_classifier.py` to adjust classification rules.

### New Output Formats
Extend `src/formatters/report_formatter.py` to support additional export formats.

## Troubleshooting

### Common Issues

**"Missing required columns"**
- Ensure your CSV files have the correct column names
- The tool automatically maps common variations (e.g., "Query" → "query")

**"No data after filtering"**
- Check if your URLs are being filtered out by the URLFilter
- Review the filtering rules in `src/processors/url_filter.py`

**"Empty similarity data"**
- Ensure URLs in similarity file match those in GSC data
- Check file encoding and delimiter settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
