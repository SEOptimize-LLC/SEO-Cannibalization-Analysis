"""
Action classifier for determining recommended actions based on URL metrics
"""


class ActionClassifier:
    """Determines recommended actions for URL pairs based on metrics"""
    
    def __init__(self):
        self.similarity_threshold_high = 0.90
        self.similarity_threshold_low = 0.89
    
    def classify(self, row: dict) -> str:
        """
        Classify the recommended action for a URL pair
        
        Args:
            row: Dictionary with URL metrics including:
                - similarity_score: float
                - primary_url_indexed_queries: int
                - primary_url_clicks: int
                - secondary_url_indexed_queries: int
                - secondary_url_clicks: int
                - primary_url_impressions: int
                - secondary_url_impressions: int
        
        Returns:
            Recommended action string
        """
        primary_queries = row.get('primary_url_indexed_queries', 0)
        secondary_queries = row.get('secondary_url_indexed_queries', 0)
        primary_clicks = row.get('primary_url_clicks', 0)
        secondary_clicks = row.get('secondary_url_clicks', 0)
        similarity = row.get('similarity_score', 0)
        
        # Case 1: Remove - both URLs have 0 clicks and 0 indexed keywords
        if (primary_clicks == 0 and secondary_clicks == 0 and 
            primary_queries == 0 and secondary_queries == 0):
            return 'Remove'
        
        # Case 2: Merge - similarity >= 90% with at least 1 click and 1 keyword
        if (similarity >= self.similarity_threshold_high and
            primary_queries >= 1 and secondary_queries >= 1 and
            primary_clicks >= 1 and secondary_clicks >= 1):
            return 'Merge'
        
        # Case 3: Redirect - similarity <= 89% with at least 1 keyword and 1 click
        if (similarity <= self.similarity_threshold_low and
            primary_queries >= 1 and secondary_queries >= 1 and
            primary_clicks >= 1 and secondary_clicks >= 1):
            return 'Redirect'
        
        # Case 4: Internal Link - similarity <= 89% with significant traffic
        if (similarity <= self.similarity_threshold_low and
            ((primary_queries > 1 and primary_clicks > 1) or 
             (secondary_queries > 1 and secondary_clicks > 1))):
            return 'Internal Link'
        
        # Case 5: Optimize - similarity <= 89% with low keywords/clicks but decent impressions
        total_impressions = (row.get('primary_url_impressions', 0) + 
                           row.get('secondary_url_impressions', 0))
        if (similarity <= self.similarity_threshold_low and
            total_impressions > 100 and
            primary_queries <= 5 and secondary_queries <= 5 and
            primary_clicks <= 5 and secondary_clicks <= 5):
            return 'Optimize'
        
        # Default: False Positive
        return 'False Positive'
