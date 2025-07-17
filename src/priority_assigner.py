"""
Priority assignment module - assigns High/Medium/Low priority based on traffic potential
"""


class PriorityAssigner:
    """Assigns priority based on traffic potential"""
    
    @staticmethod
    def assign_priority(row):
        """Assign High/Medium/Low priority based on traffic share"""
        if row['recommended_action'] in ['Remove', 'False Positive']:
            return 'Low'
        
        total_keywords = row['primary_url_indexed_queries'] + \
            row['secondary_url_indexed_queries']
        total_clicks = row['primary_url_clicks'] + row['secondary_url_clicks']
        
        # Calculate traffic potential score
        traffic_score = total_keywords + total_clicks
        
        # Determine thresholds based on distribution
        if traffic_score >= 100:
            return 'High'
        elif traffic_score >= 10:
            return 'Medium'
        else:
            return 'Low'
