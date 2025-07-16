"""
Enhanced Consolidation Analysis Module
Handles large datasets with 2,000+ URLs efficiently
"""

import pandas as pd
from typing import Dict, List
import hashlib


class EnhancedConsolidationAnalyzer:
    """Enhanced consolidation analysis for large datasets"""

    def __init__(self, max_urls: int = 2000, min_overlap: int = 1):
        self.max_urls = max_urls
        self.min_overlap = min_overlap

    def analyze_url_consolidation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze URL consolidation opportunities for large datasets
        
        Args:
            df: DataFrame with query-page performance data
            
        Returns:
            Dictionary with URL-level consolidation analysis
        """
        print(f"Processing {len(df)} rows...")
        
        # Step 1: Consolidate URL metrics efficiently
        url_consolidated = self._consolidate_url_metrics_fast(df)
        print(f"Found {len(url_consolidated)} unique URLs")
        
        # Step 2: Limit to manageable number if needed
        if len(url_consolidated) > self.max_urls:
            print(f"Limiting to top {self.max_urls} URLs by traffic...")
            url_consolidated = self._limit_top_urls(url_consolidated, self.max_urls)
        
        # Step 3: Calculate keyword overlap efficiently
        url_overlap_matrix = self._calculate_overlap_efficient(url_consolidated)
        print(f"Found {len(url_overlap_matrix)} URL pairs with overlap")
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations_fast(
            url_consolidated, url_overlap_matrix
        )
        
        # Step 5: Summary
        summary = self._create_summary(recommendations)
        
        return {
            'url_metrics': url_consolidated,
            'keyword_overlap': url_overlap_matrix,
            'consolidation_recommendations': recommendations,
            'summary': summary
        }

    def _consolidate_url_metrics_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast URL consolidation using vectorized operations"""
        # Group by URL and aggregate
        consolidated = df.groupby('page').agg({
            'query': lambda x: list(x.unique()),
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        consolidated.columns = [
            'page', 'queries', 'total_clicks', 'total_impressions', 'avg_position'
        ]
        
        # Add calculated fields
        consolidated['num_queries'] = consolidated['queries'].apply(len)
        consolidated['ctr'] = (
            consolidated['total_clicks'] / 
            consolidated['total_impressions'] * 100
        ).round(2)
        
        return consolidated

    def _limit_top_urls(self, url_df: pd.DataFrame, max_urls: int) -> pd.DataFrame:
        """Limit to top URLs by traffic for performance"""
        return url_df.nlargest(max_urls, 'total_clicks')

    def _calculate_overlap_efficient(self, url_df: pd.DataFrame) -> pd.DataFrame:
        """Efficient keyword overlap calculation for large datasets"""
        # Create URL->queries mapping
        url_queries = dict(zip(url_df['page'], url_df['queries']))
        
        overlap_data = []
        urls = list(url_queries.keys())
        
        # Use set operations for speed
        for i, url1 in enumerate(urls):
            queries1 = set(url_queries[url1])
            
            for url2 in urls[i+1:]:
                queries2 = set(url_queries[url2])
                
                # Fast intersection
                intersection = queries1 & queries2
                if len(intersection) < self.min_overlap:
                    continue
                
                # Fast union and similarity
                union_size = len(queries1 | queries2)
                if union_size == 0:
                    continue
                
                jaccard_similarity = round((len(intersection) / union_size) * 100, 2)
                
                overlap_data.append({
                    'url1': url1,
                    'url2': url2,
                    'overlap_count': len(intersection),
                    'url1_total_queries': len(queries1),
                    'url2_total_queries': len(queries2),
                    'jaccard_similarity': jaccard_similarity
                })
        
        return pd.DataFrame(overlap_data)

    def _generate_recommendations_fast(self, url_df: pd.DataFrame, 
                                     overlap_matrix: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations efficiently"""
        if overlap_matrix.empty:
            return pd.DataFrame()
        
        # Create lookup
        url_lookup = dict(zip(url_df['page'], url_df.to_dict('records')))
        
        recommendations = []
        
        for _, row in overlap_matrix.iterrows():
            url1, url2 = row['url1'], row['url2']
            
            # Get metrics
            metrics1 = url_lookup[url1]
            metrics2 = url_lookup[url2]
            
            # Determine primary/secondary
            if metrics1['total_clicks'] >= metrics2['total_clicks']:
                primary, secondary = url1, url2
                primary_metrics, secondary_metrics = metrics1, metrics2
            else:
                primary, secondary = url2, url1
                primary_metrics, secondary_metrics = metrics2, metrics1
            
            # Apply consolidation logic
            similarity = row['jaccard_similarity']
            consolidation_type = self._get_consolidation_type(
                similarity, secondary_metrics, primary_metrics
            )
            
            priority = self._get_priority(secondary_metrics['total_clicks'])
            
            recommendations.append({
                'primary_page': primary,
                'primary_page_indexed_keywords': primary_metrics['num_queries'],
                'primary_page_clicks': primary_metrics['total_clicks'],
                'secondary_page': secondary,
                'secondary_page_indexed_keywords': secondary_metrics['num_queries'],
                'secondary_page_clicks': secondary_metrics['total_clicks'],
                'similarity_score': similarity,
                'number_keyword_overlaping': row['overlap_count'],
                'consolidation_type': consolidation_type,
                'priority': priority
            })
        
        return pd.DataFrame(recommendations)

    def _get_consolidation_type(self, similarity: float, 
                              secondary: dict, primary: dict) -> str:
        """Determine consolidation type based on exact criteria"""
        if similarity >= 90:
            if secondary['num_queries'] >= 100:
                return 'Optimize'
            elif (abs(secondary['total_clicks'] - primary['total_clicks']) < 
                  (primary['total_clicks'] * 0.2) and
                  abs(secondary['num_queries'] - primary['num_queries']) < 
                  (primary['num_queries'] * 0.2)):
                return 'Merge'
            elif secondary['total_clicks'] == 0 and secondary['num_queries'] == 0:
                return 'Remove'
            elif secondary['total_clicks'] < (primary['total_clicks'] + 
                                            secondary['total_clicks']) * 0.2:
                return 'Redirect'
            else:
                return 'Redirect'
        else:
            return 'Internal Link'

    def _get_priority(self, secondary_clicks: int) -> str:
        """Determine priority based on traffic"""
        if secondary_clicks > 1000:
            return 'High'
        elif secondary_clicks > 100:
            return 'Medium'
        else:
            return 'Low'

    def _create_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics"""
        if recommendations.empty:
            return {'total_recommendations': 0}
        
        return {
            'total_recommendations': len(recommendations),
            'optimize': len(recommendations[recommendations['consolidation_type'] == 'Optimize']),
            'merge': len(recommendations[recommendations['consolidation_type'] == 'Merge']),
            'remove': len(recommendations[recommendations['consolidation_type'] == 'Remove']),
            'redirect': len(recommendations[recommendations['consolidation_type'] == 'Redirect']),
            'internal_link': len(recommendations[recommendations['consolidation_type'] == 'Internal Link']),
            'high_priority': len(recommendations[recommendations['priority'] == 'High']),
            'medium_priority': len(recommendations[recommendations['priority'] == 'Medium']),
            'low_priority': len(recommendations[recommendations['priority'] == 'Low'])
        }
