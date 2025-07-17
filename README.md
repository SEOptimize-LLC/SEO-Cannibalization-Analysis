# SEO Cannibalization Analysis Tool v2.0

A comprehensive Python-based tool for analyzing keyword cannibalization issues using Google Search Console data and semantic similarity reports. This tool helps identify when multiple pages on your website compete for the same keywords, potentially diluting your SEO performance.

## 🚀 Features

- **Advanced Data Processing**: Handles multiple file formats (CSV with various delimiters, Excel)
- **Smart URL Filtering**: Automatically excludes irrelevant URLs (parameters, legal pages, archives, etc.)
- **Semantic Analysis Integration**: Combines GSC data with semantic similarity scores
- **Actionable Recommendations**: Provides specific actions (Merge, Redirect, Optimize, etc.)
- **Priority-based Reporting**: Identifies high-impact opportunities based on traffic potential
- **Modular Architecture**: Easy to extend and customize

## 📋 Requirements

- Python 3.8 or higher
- Required packages (see `requirements.txt`):
  - pandas>=2.0.0
  - numpy>=1.24.0
  - openpyxl>=3.1.0
  - pyyaml>=6.0
  - click>=8.1.0
  - tqdm>=4.65.0

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis.git
cd SEO-Cannibalization-Analysis

Install the package:

bashpip install -e .
Or install dependencies directly:
bashpip install -r requirements.txt
📊 Required Data Files
1. Google Search Console Report
Export from GSC with these columns:

query: Search query
page: URL of the page
clicks: Number of clicks
impressions: Number of impressions
position: Average position

2. Semantic Similarity Report
CSV file with at least these columns:

Address or URL: Primary URL
Closest Semantically Similar Address: Secondary URL
Semantic Similarity Score: Similarity score (0-1)

🚀 Usage
Command Line Interface
bash# Basic usage
python main.py -g gsc_report.xlsx -s similarity_report.csv

# With custom output file
python main.py -g gsc_report.xlsx -s similarity_report.csv -o results.csv

# With custom config
python main.py -g gsc_report.xlsx -s similarity_report.csv -c my_config.yaml
Python API
pythonfrom main import SEOCannibalizationTool

# Initialize the tool
tool = SEOCannibalizationTool()

# Run analysis
results = tool.run_analysis(
    gsc_file='path/to/gsc_report.xlsx',
    similarity_file='path/to/similarity_report.csv',
    output_file='results.csv'
)
⚙️ Configuration
Edit config.yaml to customize URL filtering and analysis parameters:
yamlurl_filters:
  excluded_parameters: ["?", "=", "#"]
  excluded_pages: ["privacy-policy", "terms-of-service"]
  excluded_patterns: ["/page/\\d+/", "/\\d{4}/\\d{2}/\\d{2}"]

analysis:
  similarity_thresholds:
    high: 0.90
    medium: 0.89
  priority_percentiles:
    high: 70
    medium: 30
📈 Understanding the Output
The tool generates a CSV with these columns:

URL Information:

primary_url: Main URL being analyzed
secondary_url: Similar URL that might be cannibalizing


Performance Metrics:

*_indexed_queries: Number of unique queries ranking
*_clicks: Total clicks received
*_impressions: Total impressions


Analysis Results:

similarity_score: Semantic similarity (0-1)
recommended_action: Specific action to take
priority: High/Medium/Low based on traffic impact



Recommended Actions Explained

Remove: Both URLs have no traffic - consider removing
Merge: Very similar content (≥90%) - combine into one page
Redirect: Similar content with some traffic - redirect to stronger page
Internal Link: Different enough to keep - add contextual links
Optimize: Low traffic but high impressions - improve content/CTR
False Positive: No action needed

🎯 Best Practices

Data Quality:

Use at least 3-6 months of GSC data
Include both mobile and desktop data
Ensure semantic similarity analysis is up-to-date


Implementation:

Start with High priority items
Test changes on staging first
Monitor rankings after changes
Allow 4-8 weeks for results


Regular Audits:

Run analysis quarterly
Track improvements over time
Update URL filters as needed



🐛 Troubleshooting
Common Issues

"File must have at least 3 columns" Error:

Check CSV delimiter (comma vs semicolon)
Verify file encoding (UTF-8 recommended)


Missing Data in Output:

Check if URLs were filtered out
Verify URL format consistency between files


No High Priority Items:

May indicate good SEO health
Check if traffic thresholds are too high



🤝 Contributing
We welcome contributions! Please:

Fork the repository
Create a feature branch
Add tests for new functionality
Submit a pull request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Support
For questions or issues:

Open an issue on GitHub
Check existing issues for solutions
Review the documentation


Built with ❤️ by SEOptimize LLC
