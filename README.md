## __SEO Cannibalization Analysis Tool - Overview__

This is a sophisticated __Streamlit-based web application__ that analyzes __keyword cannibalization issues__ using Google Search Console (GSC) data. The tool helps SEO professionals identify when multiple pages on a website compete for the same keywords, potentially diluting search performance.

## __Core Purpose & Functionality__

### __Primary Goal__

Identify and resolve keyword cannibalization issues to improve organic search performance by:

- Detecting when multiple URLs target identical keywords
- Providing actionable consolidation recommendations
- Prioritizing fixes based on traffic impact

### __Key Features__

1. __Data Processing & Cleaning__

   - Accepts Google Search Console CSV exports
   - Normalizes column names automatically
   - Filters out branded keywords, spam, and invalid data
   - Handles various CSV formats and delimiters

2. __Cannibalization Detection__

   - Identifies queries where multiple pages receive clicks

   - Uses statistical thresholds (minimum 10% click distribution)

   - Calculates cannibalization scores based on:

     - Number of competing pages
     - Click distribution entropy
     - Traffic volume potential

3. __URL Consolidation Analysis__

   - Analyzes semantic similarity between URLs
   - Provides specific recommendations: Merge, Redirect, Optimize, Internal Link, Monitor
   - Prioritizes actions based on traffic recovery potential

4. __Interactive Dashboard__

   - Real-time analysis with progress indicators
   - Filterable recommendations by action type and priority
   - Downloadable CSV reports
   - Visual metrics and summaries

## __Technical Architecture__

### __Input Requirements__

- __Primary__: Google Search Console CSV with columns: `page`, `query`, `clicks`, `impressions`, `position`
- __Optional__: Semantic similarity CSV for enhanced analysis

### __Analysis Pipeline__

1. __Data Cleaning__: Removes invalid URLs, parameters, subdomains
2. __Brand Filtering__: Excludes branded keyword variations
3. __Cannibalization Detection__: Identifies competing pages
4. __URL Analysis__: Evaluates consolidation opportunities
5. __Priority Assignment__: Ranks fixes by impact potential

### __Output Deliverables__

- __Cannibalization Summary__: Overview of affected queries
- __URL Consolidation Report__: Specific merge/redirect recommendations
- __Priority Matrix__: Ranked list of issues by impact and ease of fix
- __Detailed Metrics__: Click distribution, traffic estimates, semantic similarity scores

## __Use Cases__

- __Content Audits__: Identify redundant competing content
- __Site Migrations__: Detect cannibalization before/after changes
- __Content Strategy__: Plan new content to avoid keyword conflicts
- __Performance Optimization__: Improve underperforming pages
- __Competitive Analysis__: Understand internal competition patterns

## __Advanced Capabilities__

- __Semantic Analysis__: Uses pre-calculated URL similarity scores for enhanced recommendations
- __Traffic Recovery Estimation__: Predicts potential traffic gains from consolidation
- __Multi-layered Filtering__: Brand exclusion, click thresholds, statistical validation
- __Export Functionality__: Generates actionable CSV reports for implementation

The tool is designed for SEO professionals, content strategists, and digital marketers who need to optimize their website's search performance by eliminating internal keyword competition.
