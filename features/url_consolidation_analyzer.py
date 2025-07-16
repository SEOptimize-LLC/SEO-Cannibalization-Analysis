"""
URL Consolidation Analyzer
Provides URL-level analysis with keyword overlap, semantic similarity, and comprehensive metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict

class URLConsolidationAnalyzer:
    """Enhanced URL consolidation analysis with multiple metrics"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def analyze_url_consolidation(self, df: pd.DataFrame, embeddings_df: pd.DataFrame = None) -> Dict:
        """
        Analyze URL consolidation opportunities with enhanced metrics
        
        Args:
            df: DataFrame with query-page performance data
            embeddings_df: Optional DataFrame with URL embeddings for semantic similarity
            
        Returns:
            Dictionary with URL-level consolidation analysis
        """
        # Calculate URL-level metrics
        url_metrics = self._calculate_url_metrics(df)
        
        # Calculate keyword overlap between URL pairs
        url_overlap_matrix = self._calculate_keyword_overlap(df)
        
        # Calculate semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(df, embeddings_df)
        
        # Generate consolidation recommendations
        recommendations = self._generate_consolidation_recommendations(
            df, url_metrics, url_overlap_matrix, semantic_similarity
        )
        
        # Create summary statistics
        summary = self._create_consolidation_summary(recommendations)
        
        return {
            'recommendations': recommendations,
            'summary': summary,
            'embeddings_used': embeddings_df is not None
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
        
        # Calculate additional metrics
        url_metrics['num_queries'] = url_metrics['queries'].apply(len)
        url_metrics['ctr'] = (url_metrics['total_clicks'] / url_metrics['total_impressions'] * 100).round(2)
        url_metrics['avg_position'] = url_metrics['avg_position'].round(2)
        
        return url_metrics
    
    def _calculate_keyword_overlap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate keyword overlap between URL pairs - optimized version"""
        # Create URL-query mapping with performance filtering
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        url_clicks = df.groupby('page')['clicks'].sum().to_dict()
        
        # Filter URLs with significant traffic to reduce pairs
        min_clicks_threshold = 5
        significant_urls = [url for url, clicks in url_clicks.items() if clicks >= min_clicks_threshold]
        
        if len(significant_urls) < 2:
            # If not enough significant URLs, use top URLs by clicks
            top_urls = sorted(url_clicks.items(), key=lambda x: x[1], reverse=True)[:50]
            significant_urls = [url for url, _ in top_urls]
        
        # Limit to prevent excessive computation
        max_urls = 100
        if len(significant_urls) > max_urls:
            significant_urls = significant_urls[:max_urls]
        
        # Pre-calculate query performance for efficiency
        query_performance = df.groupby(['query', 'page']).agg({
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        overlap_data = []
        
        # Use efficient set operations
        for i, url1 in enumerate(significant_urls):
            queries1 = url_queries[url1]
            if not queries1:
                continue
                
            for url2 in significant_urls[i+1:]:
                queries2 = url_queries[url2]
                if not queries2:
                    continue
                
                # Calculate overlap efficiently
                intersection = queries1 & queries2
                if len(intersection) == 0:
                    continue
                
                union = queries1 | queries2
                
                overlap_count = len(intersection)
                total_queries1 = len(queries1)
                total_queries2 = len(queries2)
                
                overlap_percentage1 = (overlap_count / total_queries1 * 100) if total_queries1 > 0 else 0
                overlap_percentage2 = (overlap_count / total_queries2 * 100) if total_queries2 > 0 else 0
                jaccard_similarity = (len(intersection) / len(union) * 100) if len(union) > 0 else 0
                
                # Calculate combined performance efficiently
                overlap_mask = query_performance['query'].isin(intersection)
                url1_mask = overlap_mask & (query_performance['page'] == url1)
                url2_mask = overlap_mask & (query_performance['page'] == url2)
                
                combined_clicks = query_performance.loc[url1_mask, 'clicks'].sum() + query_performance.loc[url2_mask, 'clicks'].sum()
                combined_impressions = query_performance.loc[url1_mask, 'impressions'].sum() + query_performance.loc[url2_mask, 'impressions'].sum()
                
                # Only include pairs with meaningful overlap
                if jaccard_similarity >= 5:  # Minimum 5% overlap
                    overlap_data.append({
                        'url1': url1,
                        'url2': url2,
                        'overlap_queries': list(intersection),
                        'overlap_count': overlap_count,
                        'url1_total_queries': total_queries1,
                        'url2_total_queries': total_queries2,
                        'overlap_percentage_url1': round(overlap_percentage1, 2),
                        'overlap_percentage_url2': round(overlap_percentage2, 2),
                        'jaccard_similarity': round(jaccard_similarity, 2),
                        'combined_overlap_clicks': combined_clicks,
                        'combined_overlap_impressions': combined_impressions
                    })
        
        return pd.DataFrame(overlap_data)
    
    def _calculate_semantic_similarity(self, df: pd.DataFrame, embeddings_df: pd.DataFrame = None) -> Dict:
        """Calculate semantic similarity between URLs"""
        similarity_scores = {}
        
        if embeddings_df is not None:
            # Use provided embeddings for enhanced similarity
            return self._calculate_embedding_similarity(embeddings_df)
        else:
            # Use basic TF-IDF similarity based on queries
            return self._calculate_tfidf_similarity(df)
    
    def _calculate_tfidf_similarity(self, df: pd.DataFrame) -> Dict:
        """Calculate TF-IDF similarity based on queries"""
        url_queries = df.groupby('page')['query'].apply(lambda x: ' '.join(x)).to_dict()
        
        if len(url_queries) < 2:
            return {}
        
        # Prepare documents
        urls = list(url_queries.keys())
        documents = [url_queries[url] for url in urls]
        
        # Calculate TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create similarity dictionary
            similarity_scores = {}
            for i, url1 in enumerate(urls):
                for j, url2 in enumerate(urls):
                    if i < j:  # Only store unique pairs
                        similarity_scores[(url1, url2)] = round(similarity_matrix[i][j] * 100, 2)
            
            return similarity_scores
        except:
            # Fallback to simple keyword overlap similarity
            return self._calculate_simple_similarity(df)
    
    def _calculate_simple_similarity(self, df: pd.DataFrame) -> Dict:
        """Calculate simple similarity based on keyword overlap"""
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        similarity_scores = {}
        
        urls = list(url_queries.keys())
        for i, url1 in enumerate(urls):
            for j, url2 in enumerate(urls):
                if i < j:
                    queries1 = url_queries[url1]
                    queries2 = url_queries[url2]
                    
                    intersection = queries1.intersection(queries2)
                    union = queries1.union(queries2)
                    
                    similarity = (len(intersection) / len(union) * 100) if len(union) > 0 else 0
                    similarity_scores[(url1, url2)] = round(similarity, 2)
        
        return similarity_scores
    
    def _calculate_embedding_similarity(self, embeddings_df: pd.DataFrame) -> Dict:
        """Calculate similarity using provided embeddings"""
        similarity_scores = {}
        
        # Ensure embeddings_df has URL column
        if 'url' not in embeddings_df.columns:
            return {}
        
        # Create URL to embedding mapping
        url_embeddings = {}
        for _, row in embeddings_df.iterrows():
            url = row['url']
            embeddings = row.drop('url').values
            url_embeddings[url] = embeddings
        
        # Calculate cosine similarity for all pairs
        urls = list(url_embeddings.keys())
        for i, url1 in enumerate(urls):
            for j, url2 in enumerate(urls):
                if i < j and url1 in url_embeddings and url2 in url_embeddings:
                    vec1 = url_embeddings[url1].reshape(1, -1)
                    vec2 = url_embeddings[url2].reshape(1, -1)
                    
                    similarity = cosine_similarity(vec1, vec2)[0][0]
                    similarity_scores[(url1, url2)] = round(similarity * 100, 2)
        
        return similarity_scores
    
    def _generate_consolidation_recommendations(self, df: pd.DataFrame, 
                                              url_metrics: pd.DataFrame,
                                              overlap_matrix: pd.DataFrame,
                                              semantic_similarity: Dict) -> pd.DataFrame:
        """Generate consolidation recommendations"""
        recommendations = []
        
        for _, overlap in overlap_matrix.iterrows():
            url1, url2 = overlap['url1'], overlap['url2']
            
            # Get URL metrics
            url1_metrics = url_metrics[url_metrics['page'] == url1].iloc[0]
            url2_metrics = url_metrics[url_metrics['page'] == url2].iloc[0]
            
            # Determine primary URL (higher clicks)
            if url1_metrics['total_clicks'] >= url2_metrics['total_clicks']:
                primary_url, secondary_url = url1, url2
                primary_metrics, secondary_metrics = url1_metrics, url2_metrics
                primary_overlap_pct = overlap['overlap_percentage_url1']
                secondary_overlap_pct = overlap['overlap_percentage_url2']
            else:
                primary_url, secondary_url = url2, url1
                primary_metrics, secondary_metrics = url2_metrics, url1_metrics
                primary_overlap_pct = overlap['overlap_percentage_url2']
                secondary_overlap_pct = overlap['overlap_percentage_url1']
            
            # Calculate consolidation metrics
            total_combined_clicks = primary_metrics['total_clicks'] + secondary_metrics['total_clicks']
            total_combined_impressions = primary_metrics['total_impressions'] + secondary_metrics['total_impressions']
            
            # Traffic recovery estimate (70% of secondary URL clicks)
            potential_recovery = int(secondary_metrics['total_clicks'] * 0.7)
            
            # Semantic similarity score
            semantic_score = semantic_similarity.get(
                (primary_url, secondary_url), 
                semantic_similarity.get((secondary_url, primary_url), 0)
            )
            
            # Determine recommendation type and priority
            recommendation_type, priority = self._determine_recommendation(
                overlap, semantic_score, potential_recovery
            )
            
            recommendations.append({
                'primary_url': primary_url,
                'secondary_url': secondary_url,
                'action': recommendation_type,
                'priority': priority,
                'keyword_overlap_count': overlap['overlap_count'],
                'keyword_overlap_percentage': (primary_overlap_pct + secondary_overlap_pct) / 2,
                'jaccard_similarity': overlap['jaccard_similarity'],
                'semantic_similarity': semantic_score,
                'primary_clicks': primary_metrics['total_clicks'],
                'secondary_clicks': secondary_metrics['total_clicks'],
                'combined_clicks': total_combined_clicks,
                'combined_impressions': total_combined_impressions,
                'potential_recovery': potential_recovery,
                'shared_keywords': overlap['overlap_queries']
            })
        
        return pd.DataFrame(recommendations)
    
    def _determine_recommendation(self, overlap: pd.Series, 
                                semantic_score: float, recovery: int) -> Tuple[str, str]:
        """Determine recommendation type and priority"""
        avg_overlap = (overlap['overlap_percentage_url1'] + overlap['overlap_percentage_url2']) / 2
        
        # Determine action type
        if avg_overlap >= 60 and semantic_score >= 50 and recovery > 100:
            action = 'Merge'
        elif avg_overlap >= 40 and recovery > 50:
            action = 'Redirect'
        elif avg_overlap >= 20 and semantic_score >= 30:
            action = 'Optimize'
        elif avg_overlap >= 10:
            action = 'Internal Link'
        elif avg_overlap >= 5:
            action = 'Monitor'
        else:
            action = 'False Positive'
        
        # Determine priority
        if recovery > 500:
            priority = 'High'
        elif recovery > 100:
            priority = 'Medium'
        else:
            priority = 'Low'
        
        return action, priority
    
    def _create_consolidation_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics for consolidation recommendations"""
        if recommendations.empty:
            return {
                'total_pairs': 0,
                'total_potential_recovery': 0,
                'priorities': {},
                'actions': {}
            }
        
        # Count actions
        actions = recommendations['action'].value_counts().to_dict()
        
        # Count priorities
        priorities = recommendations['priority'].value_counts().to_dict()
        
        return {
            'total_pairs': len(recommendations),
            'total_potential_recovery': recommendations['potential_recovery'].sum(),
            'priorities': priorities,
            'actions': actions
        }
