"""
Priority assigner for determining High/Medium/Low priority based on traffic
"""


class PriorityAssigner:
    """Assigns priority levels based on traffic metrics"""
    
    def __init__(self):
        self.high_threshold = 0.20  # 20%
        self.low_threshold = 0.01   # 1%
    
    def assign(self, row: dict, total_clicks: int) -> str:
        """
        Assign priority based on traffic metrics
        
        Args:
            row: Dictionary with URL metrics
            total_clicks: Total clicks across all URLs for normalization
        
        Returns:
            Priority string: 'High', 'Medium', or 'Low'
        """
        if total_clicks == 0:
            return 'Low'
        
        # Calculate combined clicks for the URL pair
        combined_clicks = (
            row.get('primary_url_clicks', 0) + 
            row.get('secondary_url_clicks', 0)
        )
        
        # Calculate percentage of total traffic
        traffic_percentage = combined_clicks / total_clicks
        
        # Determine priority based on thresholds
        if traffic_percentage >= self.high_threshold:
            return 'High'
        elif traffic_percentage >= self.low_threshold:
            return 'Medium'
        else:
            return 'Low'
