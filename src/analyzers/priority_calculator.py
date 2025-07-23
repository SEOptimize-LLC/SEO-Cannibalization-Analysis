import numpy as np
import logging

logger = logging.getLogger(__name__)


class PriorityCalculator:
    """Calculates priority levels based on traffic potential"""

    def __init__(self, config):
        self.config = config
        # Support both new and old config structure
        if hasattr(config, 'analysis'):
            # New config structure
            self.high_percentile = config.analysis.priority_percentiles.high
            self.medium_percentile = config.analysis.priority_percentiles.medium
        else:
            # Fallback to old structure
            self.high_percentile = 75
            self.medium_percentile = 40

    def calculate_priorities(self, data):
        """Calculate priority levels for each URL pair"""
        logger.info("Calculating priorities...")

        # Filter out Remove and False Positive actions
        priority_actions = ['Internal Link', 'Optimize', 'Redirect', 'Merge']
        priority_mask = data['recommended_action'].isin(priority_actions)

        # Calculate combined traffic score
        data['traffic_score'] = (
            data['primary_url_clicks'] + data['secondary_url_clicks'] +
            (data['primary_url_impressions'] + data['secondary_url_impressions']) * 0.1 +
            (data['primary_url_indexed_queries'] + data['secondary_url_indexed_queries']) * 5
        )

        # Initialize priority column
        data['priority'] = 'N/A'

        if priority_mask.sum() > 0:
            # Calculate percentiles only for actionable items
            traffic_scores = data.loc[priority_mask, 'traffic_score']

            # Calculate percentile thresholds
            high_threshold = np.percentile(traffic_scores, self.high_percentile)
            medium_threshold = np.percentile(traffic_scores, self.medium_percentile)

            # Assign priorities
            data.loc[
                priority_mask & (data['traffic_score'] >= high_threshold),
                'priority'
            ] = 'High'
            data.loc[
                priority_mask & (data['traffic_score'] >= medium_threshold) &
                (data['traffic_score'] < high_threshold),
                'priority'
            ] = 'Medium'
            data.loc[
                priority_mask & (data['traffic_score'] < medium_threshold),
                'priority'
            ] = 'Low'

        # Log priority distribution
        priority_counts = data[data['priority'] != 'N/A']['priority'].value_counts()
        logger.info("Priority distribution:")
        for priority, count in priority_counts.items():
            logger.info(f"  {priority}: {count}")

        # Drop the temporary traffic_score column
        data = data.drop('traffic_score', axis=1)

        return data
