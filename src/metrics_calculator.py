"""
Metrics calculation module - calculates indexed queries, clicks, impressions per URL
"""


class MetricsCalculator:
    """Calculates metrics per URL from GSC data"""
    
    @staticmethod
    def calculate_metrics(gsc_df):
        """Calculate indexed queries, clicks, impressions per URL"""
        metrics = gsc_df.groupby('page').agg({
            'query': 'nunique',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        metrics.columns = ['page', 'indexed_queries', 'clicks', 'impressions']
        return metrics
