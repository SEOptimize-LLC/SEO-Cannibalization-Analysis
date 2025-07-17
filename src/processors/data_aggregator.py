import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataAggregator:
    """Aggregates GSC data by URL and merges with similarity data"""
    
    def __init__(self):
        self.gsc_aggregated = None
        self.merged_data = None
    
    def aggregate_gsc_data(self, gsc_data):
        """Aggregate GSC data by URL"""
        logger.info("Aggregating GSC data by URL...")
        
        # Group by page and aggregate
        self.gsc_aggregated = gsc_data.groupby('page').agg({
            'query': 'count',  # Count of unique queries
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        # Rename columns
        self.gsc_aggregated.columns = ['url', 'indexed_queries', 'total_clicks', 'total_impressions']
        
        # Round numeric values
        self.gsc_aggregated['total_clicks'] = self.gsc_aggregated['total_clicks'].round(2)
        self.gsc_aggregated['total_impressions'] = self.gsc_aggregated['total_impressions'].round(2)
        
        logger.info(f"Aggregated to {len(self.gsc_aggregated)} unique URLs")
        
        return self.gsc_aggregated
    
    def merge_with_similarity(self, similarity_data):
        """Merge similarity data with aggregated GSC data"""
        logger.info("Merging similarity and GSC data...")
        
        # Create a copy of similarity data
        merged = similarity_data.copy()
        
        # Merge primary URL data
        merged = merged.merge(
            self.gsc_aggregated,
            left_on='primary_url',
            right_on='url',
            how='left',
            suffixes=('', '_primary')
        )
        
        # Rename columns for primary URL
        merged.rename(columns={
            'indexed_queries': 'primary_url_indexed_queries',
            'total_clicks': 'primary_url_clicks',
            'total_impressions': 'primary_url_impressions'
        }, inplace=True)
        
        # Drop the extra url column
        merged.drop('url', axis=1, inplace=True)
        
        # Merge secondary URL data
        merged = merged.merge(
            self.gsc_aggregated,
            left_on='secondary_url',
            right_on='url',
            how='left',
            suffixes=('', '_secondary')
        )
        
        # Rename columns for secondary URL
        merged.rename(columns={
            'indexed_queries': 'secondary_url_indexed_queries',
            'total_clicks': 'secondary_url_clicks',
            'total_impressions': 'secondary_url_impressions'
        }, inplace=True)
        
        # Drop the extra url column
        merged.drop('url', axis=1, inplace=True)
        
        # Fill NaN values with 0 for URLs not found in GSC data
        fill_columns = [
            'primary_url_indexed_queries', 'primary_url_clicks', 'primary_url_impressions',
            'secondary_url_indexed_queries', 'secondary_url_clicks', 'secondary_url_impressions'
        ]
        
        for col in fill_columns:
            merged[col] = merged[col].fillna(0)
        
        # Convert query counts to integers
        merged['primary_url_indexed_queries'] = merged['primary_url_indexed_queries'].astype(int)
        merged['secondary_url_indexed_queries'] = merged['secondary_url_indexed_queries'].astype(int)
        
        # Format similarity score to 2 decimal places
        merged['similarity_score'] = merged['similarity_score'].round(2)
        
        self.merged_data = merged
        
        logger.info(f"Merged data contains {len(merged)} URL pairs")
        
        return merged
