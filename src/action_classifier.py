"""
Action classification module - determines recommended actions based on exact rules
"""


class ActionClassifier:
    """Determines recommended actions based on exact rules"""
    
    @staticmethod
    def classify_action(row):
        """Classify recommended action based on exact rules"""
        primary_clicks = row['primary_url_clicks']
        secondary_clicks = row['secondary_url_clicks']
        primary_keywords = row['primary_url_indexed_queries']
        secondary_keywords = row['secondary_url_indexed_queries']
        similarity = row['semantic_similarity']
        
        # Rule 1: Remove
        if (primary_clicks == 0 and secondary_clicks == 0 and 
            primary_keywords == 0 and secondary_keywords == 0):
            return 'Remove'
        
        # Rule 2: Merge (90%+ similarity)
        if (similarity >= 90 and primary_clicks >= 1 and secondary_clicks >= 1 and
            primary_keywords >= 1 and secondary_keywords >= 1):
            return 'Merge'
        
        # Rule 3: Redirect (<90% similarity)
        if (similarity < 90 and primary_clicks >= 1 and secondary_clicks >= 1 and
            primary_keywords >= 1 and secondary_keywords >= 1):
            return 'Redirect'
        
        # Rule 4: Internal Link (<89% similarity, significant traffic)
        if (similarity <= 89 and (primary_keywords > 1 or secondary_keywords > 1) and
            (primary_clicks > 1 or secondary_clicks > 1)):
            return 'Internal Link'
        
        # Rule 5: Optimize (<89% similarity, low traffic, decent impressions)
        if (similarity <= 89 and primary_clicks <= 1 and secondary_clicks <= 1 and
            (primary_keywords <= 1 or secondary_keywords <= 1) and
            (row['primary_url_impressions'] > 100 or row['secondary_url_impressions'] > 100)):
            return 'Optimize'
        
        # Default
        return 'False Positive'
