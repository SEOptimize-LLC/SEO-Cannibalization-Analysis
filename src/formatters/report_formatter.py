"""
Report formatter for creating the final output
"""

import pandas as pd
from typing import Dict


class ReportFormatter:
    """Formats the final analysis report"""
    
    def __init__(self):
        self.output_columns = [
            'primary_url',
            'primary_url_indexed_queries',
            'primary_url_clicks',
            'primary_url_impressions',
            'secondary_url',
            'secondary_url_indexed_queries',
            'secondary_url_clicks',
            'secondary_url_impressions',
            'semantic_similarity',
            'recommended_action',
            'priority'
        ]
    
    def format_report(self, analysis_results: pd.DataFrame) -> pd.DataFrame:
        """
        Format the final report with proper column ordering and naming
        
        Args:
            analysis_results: DataFrame with analysis results
            
        Returns:
            Formatted DataFrame ready for export
        """
        # Ensure all required columns exist
        for col in self.output_columns:
            if col not in analysis_results.columns:
                analysis_results[col] = None
        
        # Select and order columns
        report = analysis_results[self.output_columns].copy()
        
        # Format similarity score to 1 decimal place
        if 'semantic_similarity' in report.columns:
            report['semantic_similarity'] = report['semantic_similarity'].round(3)
        
        # Ensure proper data types
        int_columns = [
            'primary_url_indexed_queries', 'primary_url_clicks',
            'secondary_url_indexed_queries', 'secondary_url_clicks'
        ]
        for col in int_columns:
            if col in report.columns:
                report[col] = report[col].fillna(0).astype(int)
        
        return report
    
    def export_to_csv(self, report: pd.DataFrame, output_path: str) -> None:
        """Export report to CSV file"""
        report.to_csv(output_path, index=False)
    
    def get_summary_stats(self, report: pd.DataFrame) -> Dict:
        """Get summary statistics for the report"""
        return {
            'total_pairs': len(report),
            'actions': report['recommended_action'].value_counts().to_dict(),
            'priorities': report['priority'].value_counts().to_dict()
        }
