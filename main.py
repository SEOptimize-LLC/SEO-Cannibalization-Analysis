#!/usr/bin/env python3
"""
SEO Cannibalization Analysis Tool
Main entry point for the modular analysis system
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Import our modules
from src.data_loaders.gsc_loader import GSCLoader
from src.data_loaders.similarity_loader import SimilarityLoader
from src.processors.url_filter import URLFilter
from src.analyzers.action_classifier import ActionClassifier
from src.analyzers.priority_assigner import PriorityAssigner
from src.formatters.report_formatter import ReportFormatter


class SEOCannibalizationAnalyzer:
    """Main orchestrator for the SEO cannibalization analysis"""
    
    def __init__(self):
        self.gsc_loader = GSCLoader()
        self.similarity_loader = SimilarityLoader()
        self.url_filter = URLFilter()
        self.action_classifier = ActionClassifier()
        self.priority_assigner = PriorityAssigner()
        self.report_formatter = ReportFormatter()
    
    def run_analysis(self, gsc_file, similarity_file, 
                    gsc_type='csv', similarity_type='csv',
                    output_file='cannibalization_report.csv'):
        """Run the complete analysis pipeline"""
        try:
            print("Starting SEO Cannibalization Analysis...")
            
            # Step 1: Load GSC data
            print("Loading Google Search Console data...")
            gsc_df = self.gsc_loader.load(gsc_file, gsc_type)
            print(f"Loaded {len(gsc_df)} rows from GSC data")
            
            # Step 2: Load similarity data
            print("Loading semantic similarity data...")
            similarity_df = self.similarity_loader.load(similarity_file, similarity_type)
            print(f"Loaded {len(similarity_df)} URL pairs from similarity data")
            
            # Step 3: Filter URLs
            print("Filtering unwanted URLs...")
            gsc_df = self.url_filter.filter_dataframe(gsc_df)
            print(f"Filtered to {len(gsc_df)} valid URLs")
            
            # Step 4: Aggregate GSC metrics
            print("Aggregating GSC metrics...")
            gsc_metrics = self.gsc_loader.aggregate_metrics(gsc_df)
            print(f"Aggregated metrics for {len(gsc_metrics)} unique URLs")
            
            # Step 5: Validate URLs against GSC data
            print("Validating URLs against GSC data...")
            gsc_urls = set(gsc_metrics['page'])
            similarity_df = self.similarity_loader.validate_urls_against_gsc(
                similarity_df, gsc_urls
            )
            print(f"Validated {len(similarity_df)} URL pairs")
            
            # Step 6: Merge data
            print("Merging GSC and similarity data...")
            merged_df = self._merge_data(similarity_df, gsc_metrics)
            print(f"Merged data for {len(merged_df)} URL pairs")
            
            # Step 7: Calculate total clicks for priority assignment
            total_clicks = gsc_metrics['total_clicks'].sum()
            
            # Step 8: Classify actions and assign priorities
            print("Classifying actions and assigning priorities...")
            merged_df['recommended_action'] = merged_df.apply(
                self.action_classifier.classify, axis=1
            )
            merged_df['priority'] = merged_df.apply(
                lambda row: self.priority_assigner.assign(row, total_clicks), axis=1
            )
            
            # Step 9: Format final report
            print("Formatting final report...")
            final_report = self.report_formatter.format_report(merged_df)
            
            # Step 10: Export results
            print("Exporting results...")
            self.report_formatter.export_to_csv(final_report, output_file)
            
            # Step 11: Print summary
            summary = self.report_formatter.get_summary_stats(final_report)
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            print(f"Total URL pairs analyzed: {summary['total_pairs']}")
            print("\nRecommended actions:")
            for action, count in summary['actions'].items():
                print(f"   {action}: {count}")
            print("\nPriority distribution:")
            for priority, count in summary['priorities'].items():
                print(f"   {priority}: {count}")
            print("="*50)
            print(f"Report saved to: {output_file}")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    def _merge_data(self, similarity_df, gsc_metrics):
        """Merge similarity data with GSC metrics"""
        # Rename columns for consistency
        gsc_metrics = gsc_metrics.rename(columns={
            'page': 'url',
            'indexed_queries': 'indexed_queries',
            'total_clicks': 'clicks',
            'total_impressions': 'impressions'
        })
        
        # Merge primary URL data
        merged = similarity_df.merge(
            gsc_metrics,
            left_on='primary_url',
            right_on='url',
            how='left',
            suffixes=('', '_primary')
        )
        
        # Rename primary columns
        merged = merged.rename(columns={
            'indexed_queries': 'primary_url_indexed_queries',
            'clicks': 'primary_url_clicks',
            'impressions': 'primary_url_impressions'
        })
        
        # Merge secondary URL data
        merged = merged.merge(
            gsc_metrics,
            left_on='secondary_url',
            right_on='url',
            how='left',
            suffixes=('', '_secondary')
        )
        
        # Rename secondary columns
        merged = merged.rename(columns={
            'indexed_queries': 'secondary_url_indexed_queries',
            'clicks': 'secondary_url_clicks',
            'impressions': 'secondary_url_impressions'
        })
        
        # Clean up temporary columns
        columns_to_drop = [col for col in merged.columns if col.endswith(('_primary', '_secondary'))]
        merged = merged.drop(columns=columns_to_drop)
        merged = merged.drop(columns=['url'], errors='ignore')
        
        # Fill NaN values with 0 for numeric columns
        numeric_cols = [
            'primary_url_indexed_queries', 'primary_url_clicks', 'primary_url_impressions',
            'secondary_url_indexed_queries', 'secondary_url_clicks', 'secondary_url_impressions'
        ]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
        
        return merged


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='SEO Cannibalization Analysis Tool'
    )
    
    parser.add_argument('--gsc', required=True,
                        help='Path to Google Search Console file (CSV or XLSX)')
    parser.add_argument('--similarity', required=True,
                        help='Path to semantic similarity file (CSV or XLSX)')
    parser.add_argument('--output', default='cannibalization_report.csv',
                        help='Output file path (default: cannibalization_report.csv)')
    parser.add_argument('--gsc-type', choices=['csv', 'xlsx'], default='csv',
                        help='GSC file type (default: csv)')
    parser.add_argument('--similarity-type', choices=['csv', 'xlsx'], default='csv',
                        help='Similarity file type (default: csv)')
    
    args = parser.parse_args()
    
    # Validate file existence
    if not Path(args.gsc).exists():
        print(f"GSC file not found: {args.gsc}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.similarity).exists():
        print(f"Similarity file not found: {args.similarity}", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    analyzer = SEOCannibalizationAnalyzer()
    analyzer.run_analysis(
        gsc_file=args.gsc,
        similarity_file=args.similarity,
        gsc_type=args.gsc_type,
        similarity_type=args.similarity_type,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
