"""
URL Consolidation Analyzer with Fallback
Provides accurate URL-level analysis with keyword overlap and simple semantic similarity
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from collections import Counter

class URLConsolidationAnalyzer:
    """URL consolidation analysis with keyword overlap and text similarity"""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
    def analyze_url_consolidation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze URL consolidation opportunities with accurate metrics
        
        Args:
            df: DataFrame with query-page performance data
            
        Returns:
            Dictionary with URL-level consolidation analysis
        """
        # Calculate URL-level metrics
        url_metrics = self._calculate_url_metrics(df)
        
        # Calculate keyword overlap
        keyword_overlap = self._calculate_keyword_overlap(df)
        
        # Calculate simple semantic similarity
        semantic_similarity = self._calculate_simple_similarity(df)
        
        # Generate consolidation recommendations
        recommendations = self._generate_consolidation_recommendations(
            df, url_metrics, keyword_overlap, semantic_similarity
        )
        
        return {
            'url_metrics': url_metrics,
            'recommendations': recommendations,
            'summary': self._create_summary(recommendations)
        }
    
    def _calculate_url_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive metrics for each URL"""
        url_metrics = df.groupby('page').agg({
            'query': lambda x: list(x.unique()),
            'clicks': 'sum',
            'impressions': 'sum',
            'position': ['mean', 'min', 'max']
        }).reset_index()
        
        url_metrics.columns = ['page', 'queries', 'total_clicks', 'total_impressions', 
                              'avg_position', 'best_position', 'worst_position']
        
        url_metrics['num_queries'] = url_metrics['queries'].apply(len)
        url_metrics['ctr'] = (url_metrics['total_clicks'] / url_metrics['total_impressions'] * 100).round(2)
        
        return url_metrics
    
    def _calculate_keyword_overlap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate keyword overlap between URL pairs"""
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        
        overlap_data = []
        urls = list(url_queries.keys())
        
        for i, url1 in enumerate(urls):
            for url2 in urls[i+1:]:
                queries1 = url_queries[url1]
                queries2 = url_queries[url2]
                
                intersection = queries1.intersection(queries2)
                union = queries1.union(queries2)
                
                overlap_count = len(intersection)
                overlap_percentage1 = (overlap_count / len(queries1) * 100) if len(queries1) > 0 else 0
                overlap_percentage2 = (overlap_count / len(queries2) * 100) if len(queries2) > 0 else 0
                
                # Calculate combined performance
                overlap_df = df[df['query'].isin(intersection)]
                url1_overlap = overlap_df[overlap_df['page'] == url1]
                url2_overlap = overlap_df[overlap_df['page'] == url2]
                
                combined_clicks = url1_overlap['clicks'].sum() + url2_overlap['clicks'].sum()
                combined_impressions = url1_overlap['impressions'].sum() + url2_overlap['impressions'].sum()
                
                overlap_data.append({
                    'url1': url1,
                    'url2': url2,
                    'overlap_count': overlap_count,
                    'overlap_percentage1': round(overlap_percentage1, 2),
                    'overlap_percentage2': round(overlap_percentage2, 2),
                    'jaccard_index': round(len(intersection) / len(union) * 100, 2) if len(union) > 0 else 0,
                    'combined_clicks': combined_clicks,
                    'combined_impressions': combined_impressions,
                    'shared_keywords': list(intersection)
                })
        
        return pd.DataFrame(overlap_data)
    
    def _calculate_simple_similarity(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate simple text similarity without sklearn"""
        url_queries = df.groupby('page')['query'].apply(lambda x: ' '.join(x).lower()).to_dict()
        
        similarity_scores = {}
        urls = list(url_queries.keys())
        
        for i, url1 in enumerate(urls):
            similarity_scores[url1] = {}
            for url2 in urls:
                if url1 == url2:
                    similarity_scores[url1][url2] = 1.0
                else:
                    text1 = url_queries[url1]
                    text2 = url_queries[url2]
                    
                    # Simple Jaccard similarity on words
                    words1 = set(text1.split()) - self.stop_words
                    words2 = set(text2.split()) - self.stop_words
                    
                    intersection = words1.intersection(words2)
                    union = words1.union(words2)
                    
                    similarity = len(intersection) / len(union) if union else 0
                    similarity_scores[url1][url2] = similarity
        
        return similarity_scores
    
    def _generate_consolidation_recommendations(self, df: pd.DataFrame, 
                                              url_metrics: pd.DataFrame,
                                              keyword_overlap: pd.DataFrame,
                                              semantic_similarity: Dict) -> pd.DataFrame:
        """Generate accurate consolidation recommendations"""
        recommendations = []
        
        # Get URL metrics dict for easy lookup
        url_metrics_dict = url_metrics.set_index('page').to_dict('index')
        
        for _, overlap in keyword_overlap.iterrows():
            url1, url2 = overlap['url1'], overlap['url2']
            
            # Get metrics
            url1_metrics = url_metrics_dict[url1]
            url2_metrics = url_metrics_dict[url2]
            
            # Determine primary URL (higher clicks)
            if url1_metrics['total_clicks'] >= url2_metrics['total_clicks']:
                primary_url, secondary_url = url1, url2
                primary_metrics, secondary_metrics = url1_metrics, url2_metrics
            else:
                primary_url, secondary_url = url2, url1
                primary_metrics, secondary_metrics = url2_metrics, url1_metrics
            
            # Calculate semantic similarity
            semantic_sim = semantic_similarity[primary_url][secondary_url]
            
            # Calculate metrics
            total_combined_clicks = primary_metrics['total_clicks'] + secondary_metrics['total_clicks']
            potential_recovery = int(secondary_metrics['total_clicks'] * 0.7)
            
            # Determine action based on comprehensive analysis
            action = self._determine_action(
                overlap['overlap_count'],
                overlap['overlap_percentage1'],
                overlap['overlap_percentage2'],
                semantic_sim,
                primary_metrics['total_clicks'],
                secondary_metrics['total_clicks']
            )
            
            # Calculate priority
            priority = self._calculate_priority(
                potential_recovery,
                overlap['overlap_count'],
                semantic_sim
            )
            
            recommendations.append({
                'primary_url': primary_url,
                'secondary_url': secondary_url,
                'action': action,
                'keyword_overlap_count': overlap['overlap_count'],
                'keyword_overlap_percentage': max(overlap['overlap_percentage1'], overlap['overlap_percentage2']),
                'semantic_similarity': round(semantic_sim * 100, 2),
                'primary_clicks': primary_metrics['total_clicks'],
                'secondary_clicks': secondary_metrics['total_clicks'],
                'combined_clicks': total_combined_clicks,
                'potential_recovery': potential_recovery,
                'priority': priority,
                'shared_keywords': overlap['shared_keywords'],
                'primary_queries': primary_metrics['num_queries'],
                'secondary_queries': secondary_metrics['num_queries']
            })
        
        return pd.DataFrame(recommendations)
    
    def _determine_action(self, overlap_count: int, overlap_pct1: float, overlap_pct2: float,
                         semantic_sim: float, primary_clicks: int, secondary_clicks: int) -> str:
        """Determine the appropriate action for URL pair"""
        max_overlap_pct = max(overlap_pct1, overlap_pct2)
        
        # Decision tree based on overlap and performance
        if overlap_count >= 10 and max_overlap_pct >= 70:
            if secondary_clicks < primary_clicks * 0.3:
                return "Redirect"
            else:
                return "Merge"
        elif overlap_count >= 5 and max_overlap_pct >= 50:
            if semantic_sim >= 0.7:
                return "Redirect"
            else:
                return "Optimize"
        elif overlap_count >= 3 and max_overlap_pct >= 30:
            return "Internal Link"
        elif overlap_count >= 1 and max_overlap_pct >= 10:
            return "Monitor"
        else:
            return "False Positive"
    
    def _calculate_priority(self, potential_recovery: int, overlap_count: int, semantic_similarity: float) -> str:
        """Calculate priority based on potential impact"""
        score = (potential_recovery / 100) + (overlap_count / 10) + (semantic_similarity * 10)
        
        if score >= 20:
            return "High"
        elif score >= 10:
            return "Medium"
        else:
            return "Low"
    
    def _create_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics"""
        if recommendations.empty:
            return {
                'total_pairs': 0,
                'actions': {},
                'priorities': {},
                'total_potential_recovery': 0
            }
        
        actions = recommendations['action'].value_counts().to_dict()
        priorities = recommendations['priority'].value_counts().to_dict()
        
        return {
            'total_pairs': len(recommendations),
            'actions': actions,
            'priorities': priorities,
            'total_potential_recovery': recommendations['potential_recovery'].sum()
        }
