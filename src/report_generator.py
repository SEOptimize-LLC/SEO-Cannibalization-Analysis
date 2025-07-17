"""
Report generation module - creates final report with exact format
"""


class ReportGenerator:
    """Generates the final report with exact format"""
    
    @staticmethod
    def generate_report(similarity_df, gsc_metrics):
        """Create final report with exact column structure"""
        from src.action_classifier import ActionClassifier
        from src.priority_assigner import PriorityAssigner
        
        # Merge similarity data with GSC metrics
        report = similarity_df.copy()
        
        # Get metrics for primary URLs
        primary_metrics = gsc_metrics.rename(columns={
            'page': 'primary_url',
            'indexed_queries': 'primary_url_indexed_queries',
            'clicks': 'primary_url_clicks',
            'impressions': 'primary_url_impressions'
        })
        
        report = report.merge(primary_metrics, on='primary_url', how='left')
        
        # Get metrics for secondary URLs
        secondary_metrics = gsc_metrics.rename(columns={
            'page': 'secondary_url',
            'indexed_queries': 'secondary_url_indexed_queries',
            'clicks': 'secondary_url_clicks',
            'impressions': 'secondary_url_impressions'
        })
        
        report = report.merge(secondary_metrics, on='secondary_url', how='left')
        
        # Fill NaN values with 0
        numeric_cols = [
            'primary_url_indexed_queries', 'primary_url_clicks',
            'primary_url_impressions', 'secondary_url_indexed_queries',
            'secondary_url_clicks', 'secondary_url_impressions'
        ]
        report[numeric_cols] = report[numeric_cols].fillna(0)
        
        # Format similarity score to 2 decimal places
        report['semantic_similarity'] = report['semantic_similarity'].round(2)
        
        # Add classifications
        report['recommended_action'] = report.apply(
            ActionClassifier.classify_action, axis=1)
        report['priority'] = report.apply(PriorityAssigner.assign_priority, axis=1)
        
        # Reorder columns to exact specification
        final_columns = [
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
        
        return report[final_columns]
