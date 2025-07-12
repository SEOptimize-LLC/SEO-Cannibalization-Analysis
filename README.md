# SEO Cannibalization Analysis Tool

A comprehensive Python-based tool for analyzing keyword cannibalization issues using Google Search Console data. This tool helps identify when multiple pages on your website compete for the same keywords, potentially diluting your SEO performance.

## What is SEO Cannibalization?

SEO keyword cannibalization occurs when multiple URLs on a single website target the same keywords and satisfy the same user intent, causing them to compete against each other in search results[1][2]. This can lead to:

- **Reduced organic visibility** - Search engines may struggle to determine which page to rank
- **Diluted page authority** - Link equity gets spread across multiple competing pages  
- **Lower conversion rates** - Users may land on suboptimal pages for their intent
- **Confused content strategy** - Unclear targeting and messaging across pages

## Features

- **Advanced Data Processing**: Cleans and prepares Google Search Console data for analysis
- **Brand Filtering**: Removes branded keyword variations to focus on non-branded opportunities
- **Multi-layered Analysis**: Goes beyond simple overlap detection with statistical validation
- **Click Distribution Analysis**: Identifies genuine cannibalization where multiple pages receive significant traffic
- **Consolidation Recommendations**: Provides actionable insights for page merging decisions
- **Comprehensive Reporting**: Detailed output with metrics and recommendations

## Prerequisites

### Python Dependencies

Install the required Python packages:

```bash
pip install pandas numpy
```

### Google Search Console Data

You'll need a CSV export from Google Search Console with the following columns:

- `page`: The URL of the page
- `query`: The search query  
- `clicks`: Number of clicks for the page-query combination
- `impressions`: Number of impressions for the page-query combination
- `position`: Average position for the page-query combination

**Recommended data range**: 3-6 months of historical data for comprehensive analysis.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis.git
cd SEO-Cannibalization-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Export data from Google Search Console
   - Save as `dataset.csv` in the project directory
   - Ensure the CSV has the required column structure

## Configuration

Update the configuration settings in `main.py`:

```python
# Configuration Settings
FILE_LOCATION = "dataset.csv"  # Path to your GSC data file
BRAND_VARIANTS = [
    "your-brand-name",
    "yourbrand", 
    "your brand"
]  # Add your brand name variations to filter out branded queries
```

## Usage

Run the analysis:

```bash
python main.py
```

The tool will process your data through several stages:

### Stage 1: Data Cleaning and Preparation
- Loads GSC data from CSV
- Removes branded keyword variations
- Filters for keywords with multiple competing pages

### Stage 2: Cannibalization Detection  
- Identifies queries where multiple pages receive clicks
- Applies statistical thresholds to focus on significant cases
- Calculates click distribution percentages across pages

### Stage 3: Analysis and Recommendations
- Evaluates consolidation opportunities
- Generates detailed reports with metrics
- Provides actionable recommendations

## Analysis Methodology

### Click Distribution Filtering

The tool identifies genuine cannibalization by focusing on keywords where:
- At least 2 pages receive clicks for the same query
- At least 10% of total clicks are distributed among multiple pages[2]
- Excludes cases with single dominant pages (>90% click share)

### Statistical Validation

Advanced checks ensure recommendations are data-driven:
- **Traffic Volume Analysis**: Considers total traffic impact
- **Query Significance**: Evaluates keyword importance to each page
- **Performance Comparison**: Analyzes relative page performance
- **Consolidation Viability**: Assesses merger potential

## Output and Reports

The analysis generates:

- **Cannibalization Summary**: Overview of affected queries and pages
- **Detailed Query Analysis**: Per-keyword breakdown with metrics
- **Page Impact Assessment**: How cannibalization affects individual pages  
- **Consolidation Recommendations**: Specific actions to resolve issues
- **Priority Matrix**: Ranked list of issues by impact and ease of fix

## Best Practices

### Data Collection
- Use 3-6 months of data for reliable patterns[1]
- Include both mobile and desktop data
- Filter out extremely low-traffic queries
- Regularly update analysis with fresh data

### Implementation
- **Start with high-impact cases**: Focus on queries driving significant traffic
- **Content consolidation**: Merge similar pages targeting identical keywords[3]
- **301 redirects**: Properly redirect consolidated pages to preserve link equity
- **Internal linking**: Update internal link structure after consolidation

### Monitoring  
- **Track performance**: Monitor rankings after implementing changes
- **Regular audits**: Run analysis quarterly to catch new issues
- **Performance validation**: Confirm improvements in organic traffic

## Common Use Cases

1. **Content Audit**: Identify redundant or competing content
2. **Site Migration**: Detect cannibalization before/after site changes  
3. **Content Strategy**: Plan new content to avoid keyword conflicts
4. **Performance Optimization**: Improve underperforming pages
5. **Competitive Analysis**: Understand internal competition patterns

## Troubleshooting

### Common Issues

**"No cannibalization detected"**
- Check data file format and column names
- Verify sufficient data volume (recommend 1000+ queries)
- Adjust brand filtering if too aggressive

**"High false positives"**  
- Review click percentage threshold settings
- Consider seasonal variations in data
- Validate brand keyword filtering

**"Performance not improving after fixes"**
- Allow 4-8 weeks for search engine re-evaluation
- Verify 301 redirects are properly implemented
- Check for other technical SEO issues

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Include tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support:
- Open an issue on GitHub
- Review the documentation
- Check existing issues for similar problems

**Note**: This tool provides analysis and recommendations. Always validate findings manually and test changes in a staging environment before implementing on production sites.
