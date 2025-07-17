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
```

2. Install the package:
```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## 📊 Required Data Files

### 1. Google Search Console Report
Export from GSC with these columns:
- `query`: Search query
- `page`: URL of the page
- `clicks`: Number of clicks
- `impressions`: Number of impressions
- `position`: Average position

### 2. Semantic Similarity Report
CSV file with at least these columns:
- `Address` or `URL`: Primary URL
- `Closest Semantically Similar Address`: Secondary URL
- `Semantic Similarity Score`: Similarity score (0-1)

## 🚀 Usage

### Command Line Interface

```bash
# Basic usage
python main.py -g gsc_report.xlsx -s similarity_report.csv

# With custom output file
python main.py -g gsc_report.xlsx -s similarity_report.csv -o results.csv

# With custom config
python main.py -g gsc_report.xlsx -s similarity_report.csv -c my_config.yaml

# Check version
python main.py --version

# Get help
python main.py --help
```

### Python API

```python
from main import SEOCannibalizationTool

# Initialize the tool
tool = SEOCannibalizationTool()

# Run analysis
results = tool.run_analysis(
    gsc_file='path/to/gsc_report.xlsx',
    similarity_file='path/to/similarity_report.csv',
    output_file='results.csv'
)
```

## ⚙️ Configuration

Edit `config.yaml` to customize URL filtering and analysis parameters:

```yaml
url_filters:
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
```

## 📈 Understanding the Output

The tool generates a CSV with these columns:

### URL Information
- `primary_url`: Main URL being analyzed
- `secondary_url`: Similar URL that might be cannibalizing

### Performance Metrics
- `*_indexed_queries`: Number of unique queries ranking
- `*_clicks`: Total clicks received
- `*_impressions`: Total impressions

### Analysis Results
- `similarity_score`: Semantic similarity (0-1)
- `recommended_action`: Specific action to take
- `priority`: High/Medium/Low based on traffic impact

### Recommended Actions Explained

| Action | Description | When Applied |
|--------|-------------|--------------|
| **Remove** | Both URLs have no traffic - consider removing | 0 clicks, 0 queries on both URLs |
| **Merge** | Very similar content (≥90%) - combine into one page | High similarity + traffic on both |
| **Redirect** | Similar content with some traffic - redirect to stronger page | <90% similarity + some traffic |
| **Internal Link** | Different enough to keep - add contextual links | <89% similarity + good traffic |
| **Optimize** | Low traffic but high impressions - improve content/CTR | Low clicks but decent impressions |
| **False Positive** | No action needed | Doesn't meet other criteria |

## 🎯 Best Practices

### 1. Data Quality
- Use at least 3-6 months of GSC data
- Include both mobile and desktop data
- Ensure semantic similarity analysis is up-to-date

### 2. Implementation
- Start with High priority items
- Test changes on staging first
- Monitor rankings after changes
- Allow 4-8 weeks for results

### 3. Regular Audits
- Run analysis quarterly
- Track improvements over time
- Update URL filters as needed

## 🐛 Troubleshooting

### Common Issues

#### "File must have at least 3 columns" Error
- Check CSV delimiter (comma vs semicolon)
- Verify file encoding (UTF-8 recommended)
- Ensure all required columns are present

#### Missing Data in Output
- Check if URLs were filtered out
- Verify URL format consistency between files
- Review `config.yaml` exclusion rules

#### No High Priority Items
- May indicate good SEO health
- Check if traffic thresholds are too high
- Verify data completeness

### Debug Mode

Run with verbose logging:
```bash
python main.py -g data.xlsx -s similarity.csv --debug
```

## 📁 Project Structure

```
seo-cannibalization-tool/
├── src/
│   ├── __init__.py
│   ├── data_loaders/
│   │   ├── gsc_loader.py
│   │   └── similarity_loader.py
│   ├── processors/
│   │   ├── url_cleaner.py
│   │   └── data_aggregator.py
│   ├── analyzers/
│   │   ├── cannibalization_analyzer.py
│   │   └── priority_calculator.py
│   └── utils/
│       └── config.py
├── main.py
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/SEO-Cannibalization-Analysis.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues:
- 📝 Open an [issue on GitHub](https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis/issues)
- 📚 Review the [documentation](https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis/wiki)
- 💬 Check existing issues for similar problems

## 🙏 Acknowledgments

- Built with Python and love ❤️
- Thanks to all contributors
- Inspired by the SEO community

---

**Built by [SEOptimize LLC](https://seoptimize.com)**

*Making SEO analysis smarter, one tool at a time.*
