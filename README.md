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
