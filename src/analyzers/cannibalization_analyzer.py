import pandas as pd
import numpy as np
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class CannibalizationAnalyzer:
    """Analyzes URL pairs and assigns recommended actions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.similarity_high = config.similarity_thresholds.get('high', 0.90)
        self.similarity_medium = config.similarity_thresholds.get('medium', 0.89)
    
    def analyze(self, data):
        """Analyze each URL pair and assign recommended actions"""
        logger.info("Analyzing cannibalization patterns...")
        
        # Apply analysis rules
        data['recommended_action'] = data.apply(self._determine_action, axis=1)
        
        # Log action distribution
        action_counts = data['recommended_action'].value_counts()
        logger.info("Recommended actions distribution:")
        for action, count in action_counts.items():
            logger.info(f"  {action}: {count} ({count/len(data)*100:.1f}%)")
        
        return data
    
    def _determine_action(self, row):
        """Determine recommended action for a URL pair"""
        primary_queries = row['primary_url_indexed_queries']
        primary_clicks = row['primary_url_clicks']
        primary_impressions = row['primary_url_impressions']
        
        secondary_queries = row['secondary_url_indexed_queries']
        secondary_clicks = row['secondary_url_clicks']
        secondary_impressions = row['secondary_url_impressions']
        
        similarity = row['similarity_score']
        
        # Rule 1: Remove - Both URLs have 0 clicks and 0 queries
        if primary_clicks == 0 and secondary_clicks == 0 and \
           primary_queries == 0 and secondary_queries == 0:
            return 'Remove'
        
        # Rule 2: Merge - Similarity >= 90% with at least 1 click and 1 query each
        if similarity >= self.similarity_high:
            if (primary_clicks >= 1 and primary_queries >= 1) and \
               (secondary_clicks >= 1 and secondary_queries >= 1):
                return 'Merge'
        
        # Rule 3 & 4: For similarity <= 89%
        if similarity <= self.similarity_medium:
            # Both URLs have significant traffic
            if (primary_queries > 1 and primary_clicks > 1) or \
               (secondary_queries > 1 and secondary_clicks > 1):
                return 'Internal Link'
            
            # Low clicks/queries but decent impressions
            if (primary_impressions > 10 or secondary_impressions > 10) and \
               (primary_clicks <= 1 or secondary_clicks <= 1):
                return 'Optimize'
            
            # At least 1 query and 1 click each
            if (primary_queries >= 1 and primary_clicks >= 1) and \
               (secondary_queries >= 1 and secondary_clicks >= 1):
                return 'Redirect'
        
        # Default: False Positive
        return 'False Positive'
