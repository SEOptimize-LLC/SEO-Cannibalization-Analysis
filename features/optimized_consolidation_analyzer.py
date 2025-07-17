"""
Optimized URL Consolidation Analyzer
Uses pre-calculated semantic similarity data to avoid expensive computations
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class OptimizedConsolidationAnalyzer:
    """Optimized consolidation analysis using pre-calculated similarity data"""
    
    def analyze_consolidation(self, gsc_df: pd.DataFrame, similarity_df: pd.DataFrame = None) -> Dict:
        """
        Analyze URL consolidation using pre-calculated semantic similarity
        
        Args:
            gsc_df: Google Search Console data with query, page, clicks, impressions
            similarity_df: Pre-calculated similarity data with URL pairs and metrics
            
        Returns:
            Dictionary with consolidated recommendations
        """
        # Calculate URL metrics from GSC data
        url_metrics = self._calculate_url_metrics(gsc_df)
        
        # Handle case where similarity_df is None or empty
        if similarity_df is None or similarity_df.empty:
            consolidated_recommendations = pd.DataFrame()
        else:
            # Merge GSC data with similarity data
            consolidated_recommendations = self._merge_with_similarity(
                url_metrics, similarity_df
            )
        
        # Create summary
        summary = self._create_summary(consolidated_recommendations)
        
        return {
            'recommendations': consolidated_recommendations,
            'summary': summary,
            'urls_analyzed': len(url_metrics),
            'processing_method': 'pre_calculated',
            'embeddings_used': similarity_df is not None and not similarity_df.empty
        }
    
    def _calculate_url_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate URL-level metrics from GSC data"""
        url_metrics = df.groupby('page').agg({
            'query': lambda x: list(x.unique()),
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        url_metrics.columns = ['page', 'indexed_queries', 'total_clicks', 
                             'total_impressions', 'avg_position']
        
        url_metrics['num_queries'] = url_metrics['indexed_queries'].apply(len)
        url_metrics['avg_position'] = url_metrics['avg_position'].round(2)
        
        return url_metrics
    
    def _merge_with_similarity(self, url_metrics: pd.DataFrame, 
                           similarity_df: pd.DataFrame) -> pd.DataFrame:
        """Merge GSC metrics with pre-calculated similarity data"""
        recommendations = []
        
        # Process each row from similarity report
        for _, row in similarity_df.iterrows():
            # Extract URLs from similarity data
            url1 = str(row.get('url1', row.get('primary_url', '')))
            url2 = str(row.get('url2', row.get('secondary_url', '')))
            
            if not url1 or not url2:
                continue
                
            # Get GSC metrics for both URLs
            url1_data = url_metrics[url_metrics['page'] == url1]
            url2_data = url_metrics[url_metrics['page'] == url2]
            
            if url1_data.empty or url2_data.empty:
                continue
            
            url1_metrics = url1_data.iloc[0]
            url2_metrics = url2_data.iloc[0]
            
            # Determine primary URL (higher clicks)
            if url1_metrics['total_clicks'] >= url2_metrics['total_clicks']:
                primary_url, secondary_url = url1, url2
                primary_metrics, secondary_metrics = url1_metrics, url2_metrics
            else:
                primary_url, secondary_url = url2, url1
                primary_metrics, secondary_metrics = url2_metrics, url1_metrics
            
            # Calculate overlap metrics
            overlap_count = self._calculate_overlap_count(
                primary_metrics['indexed_queries'], 
                secondary_metrics['indexed_queries']
            )
            
            # Get similarity scores from pre-calculated data
            semantic_similarity = float(row.get('semantic_similarity', 0))
            keyword_overlap_percentage = float(row.get('keyword_overlap_percentage', 0))
            
            # Calculate combined metrics
            combined_clicks = primary_metrics['total_clicks'] + secondary_metrics['total_clicks']
            combined_impressions = primary_metrics['total_impressions'] + secondary_metrics['total_impressions']
            potential_recovery = int(secondary_metrics['total_clicks'] * 0.7)
            
            # Determine action based on metrics
            recommended_action = self._determine_action(
                keyword_overlap_percentage, semantic_similarity, potential_recovery
            )
            
            # Determine priority
            priority = self._determine_priority(potential_recovery, overlap_count)
            
            recommendations.append({
                'primary_url': primary_url,
                'primary_indexed_queries': primary_metrics['num_queries'],
                'primary_clicks': primary_metrics['total_clicks'],
                'primary_impressions': primary_metrics['total_impressions'],
                'primary_avg_position': primary_metrics['avg_position'],
                'secondary_url': secondary_url,
                'secondary_indexed_queries': secondary_metrics['num_queries'],
                'secondary_clicks': secondary_metrics['total_clicks'],
                'secondary_impressions': secondary_metrics['total_impressions'],
                'secondary_avg_position': secondary_metrics['avg_position'],
                'keyword_overlap_count': overlap_count,
                'keyword_overlap_percentage': keyword_overlap_percentage,
                'semantic_similarity': semantic_similarity,
                'combined_clicks': combined_clicks,
                'combined_impressions': combined_impressions,
                'potential_traffic_recovery': potential_recovery,
                'recommended_action': recommended_action,
                'priority': priority
            })
        
        return pd.DataFrame(recommendations)
    
    def _calculate_overlap_count(self, queries1: List[str], queries2: List[str]) -> int:
        """Calculate overlap count between two query lists"""
        set1 = set(queries1)
        set2 = set(queries2)
        return len(set1.intersection(set2))
    
    def _determine_action(self, overlap_pct: float, semantic_sim: float, recovery: int) -> str:
        """Determine recommended action based on metrics"""
        if overlap_pct >= 60 and recovery > 100:
            return 'Merge'
        elif overlap_pct >= 40 and recovery > 50:
            return 'Redirect'
        elif overlap_pct >= 20 and semantic_sim >= 0.7:
            return 'Optimize'
        elif overlap_pct >= 10:
            return 'Internal Link'
        elif overlap_pct >= 5:
            return 'Monitor'
        else:
            return 'False Positive'
    
    def _determine_priority(self, recovery: int, overlap_count: int) -> str:
        """Determine priority based on recovery and overlap"""
        if recovery > 500 and overlap_count > 10:
            return 'High'
        elif recovery > 100 and overlap_count > 5:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics"""
        if recommendations.empty:
            return {
                'total_pairs': 0,
                'priorities': {},
                'actions': {}
            }
        
        return {
            'total_pairs': len(recommendations),
            'priorities': recommendations['priority'].value_counts().to_dict(),
            'actions': recommendations['recommended_action'].value_counts().to_dict()
        }
