#!/usr/bin/env python3
"""
SEO Cannibalization Analysis Tool
Version: 2.0.0
Author: SEOptimize LLC
Description: Advanced tool for analyzing keyword cannibalization using GSC data and semantic similarity
"""

import logging
import click
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.utils.config import Config
from src.data_loaders.gsc_loader import GSCLoader
from src.data_loaders.similarity_loader import SimilarityLoader
from src.processors.url_cleaner import URLCleaner
from src.processors.data_aggregator import DataAggregator
from src.analyzers.cannibalization_analyzer import CannibalizationAnalyzer
from src.analyzers.priority_calculator import PriorityCalculator

__version__ = '2.0.0'
__author__ = 'SEOptimize LLC'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEOCannibalizationTool:
    """Main orchestrator for the SEO Cannibalization Analysis"""
    
    def __init__(self, config_path=None):
        self.config = Config(config_path)
        self.gsc_loader = GSCLoader()
        self.similarity_loader = SimilarityLoader()
        self.url_cleaner = URLCleaner(self.config)
        self.aggregator = DataAggregator()
        self.analyzer = CannibalizationAnalyzer(self.config)
        self.priority_calculator = PriorityCalculator(self.config)
    
    def run_analysis(self, gsc_file, similarity_file, output_file=None):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("    SEO CANNIBALIZATION ANALYSIS TOOL v" + __version__)
        print("    by " + __author__)
        print("="*60 + "\n")
        
        logger.info("Starting SEO Cannibalization Analysis")
        
        try:
            # Step 1: Load data files
            logger.info("\n[Step 1/7] Loading data files...")
            gsc_data = self.gsc_loader.load(gsc_file)
            similarity_data = self.similarity_loader.load(similarity_file)
            
            # Step 2: Clean URLs
            logger.info("\n[Step 2/7] Cleaning URLs...")
            gsc_data = self.url_cleaner.clean_dataframe(gsc_data, ['page'])
            similarity_data = self.url_cleaner.clean_dataframe(
                similarity_data, 
                ['primary_url', 'secondary_url']
            )
            
            # Step 3: Aggregate GSC data
            logger.info("\n[Step 3/7] Aggregating GSC data...")
            self.aggregator.aggregate_gsc_data(gsc_data)
            
            # Step 4: Merge data
            logger.info("\n[Step 4/7] Merging similarity and GSC data...")
            merged_data = self.aggregator.merge_with_similarity(similarity_data)
            
            # Step 5: Analyze cannibalization
            logger.info("\n[Step 5/7] Analyzing cannibalization patterns...")
            analyzed_data = self.analyzer.analyze(merged_data)
            
            # Step 6: Calculate priorities
            logger.info("\n[Step 6/7] Calculating priorities...")
            final_data = self.priority_calculator.calculate_priorities(analyzed_data)
            
            # Step 7: Generate output
            logger.info("\n[Step 7/7] Generating output report...")
            output_path = self._generate_output(final_data, output_file)
            
            # Summary statistics
            self._print_summary(final_data)
            
            logger.info(f"\n✅ Analysis complete! Results saved to: {output_path}")
            
            return final_data
            
        except Exception as e:
            logger.error(f"\n❌ Analysis failed: {e}")
            raise
    
    def _generate_output(self, data, output_file):
        """Generate the output CSV file"""
        # Reorder columns for final output
        output_columns = [
            'primary_url',
            'primary_url_indexed_queries',
            'primary_url_clicks',
            'primary_url_impressions',
            'secondary_url',
            'secondary_url_indexed_queries',
            'secondary_url_clicks',
            'secondary_url_impressions',
            'similarity_score',
            'recommended_action',
            'priority'
        ]
        
        # Ensure all columns exist
        for col in output_columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data")
        
        # Select only existing columns
        existing_columns = [col for col in output_columns if col in data.columns]
        output_data = data[existing_columns]
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'cannibalization_analysis_{timestamp}.csv'
        
        # Save to CSV
        output_path = Path(output_file)
        output_data.to_csv(output_path, index=False)
        
        # Also save a summary report
        summary_path = output_path.with_suffix('.summary.txt')
        with open(summary_path, 'w') as f:
            f.write("SEO Cannibalization Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total URL pairs analyzed: {len(data)}\n\n")
            
            f.write("Recommended Actions:\n")
            for action, count in data['recommended_action'].value_counts().items():
                f.write(f"  - {action}: {count} ({count/len(data)*100:.1f}%)\n")
            
            f.write("\nPriority Distribution (excluding Remove/False Positive):\n")
            priority_data = data[data['priority'] != 'N/A']
            for priority, count in priority_data['priority'].value_counts().items():
                f.write(f"  - {priority}: {count}\n")
        
        return output_path
    
    def _print_summary(self, data):
        """Print analysis summary to console"""
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total URL pairs analyzed: {len(data)}")
        
        print("\nRecommended Actions:")
        for action, count in data['recommended_action'].value_counts().items():
            print(f"  - {action}: {count} ({count/len(data)*100:.1f}%)")
        
        print("\nTop 10 High Priority Issues:")
        high_priority = data[data['priority'] == 'High'].head(10)
        for idx, row in high_priority.iterrows():
            print(f"\n  {idx + 1}. {row['recommended_action']} (Similarity: {row['similarity_score']:.2f})")
            print(f"     Primary: {row['primary_url'][:60]}...")
            print(f"     Secondary: {row['secondary_url'][:60]}...")
            print(f"     Impact: {row['primary_url_clicks'] + row['secondary_url_clicks']:.0f} clicks, "
                  f"{row['primary_url_indexed_queries'] + row['secondary_url_indexed_queries']} queries")


@click.command()
@click.option('--gsc-file', '-g', required=True, help='Path to Google Search Console data file')
@click.option('--similarity-file', '-s', required=True, help='Path to semantic similarity file')
@click.option('--output-file', '-o', help='Output file path (optional)')
@click.option('--config', '-c', help='Path to config.yaml file (optional)')
@click.version_option(version=__version__, prog_name='SEO Cannibalization Analysis Tool')
def main(gsc_file, similarity_file, output_file, config):
    """
    SEO Cannibalization Analysis Tool - Identify and fix keyword cannibalization issues
    
    This tool analyzes your website's keyword cannibalization by combining Google Search
    Console data with semantic similarity analysis to provide actionable recommendations.
    """
    tool = SEOCannibalizationTool(config)
    tool.run_analysis(gsc_file, similarity_file, output_file)


if __name__ == '__main__':
    main()
