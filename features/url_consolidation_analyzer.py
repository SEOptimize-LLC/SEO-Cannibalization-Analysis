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
        """Calculate keyword overlap between URL pairs"""
        # Create URL-query mapping
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        url_metrics = df.groupby('page').agg({
            'query': lambda x: len(x.unique()),
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        url_metrics.columns = ['page', 'indexed_queries', 'clicks', 'impressions']
        
        # Get all URLs
        urls = list(url_queries.keys())
        
        overlap_data = []
        
        for i, url1 in enumerate(urls):
            queries1 = url_queries[url1]
            if not queries1:
                continue
                
            for url2 in urls[i+1:]:
                queries2 = url_queries[url2]
                if not queries2:
                    continue
                
                # Calculate overlap
                intersection = queries1 & queries2
                if len(intersection) == 0:
                    continue
                
                overlap_count = len(intersection)
                
                # Calculate overlap percentages
                total_queries1 = len(queries1)
                total_queries2 = len(queries2)
                
                overlap_percentage1 = (overlap_count / total_queries1 * 100) if total_queries1 > 0 else 0
                overlap_percentage2 = (overlap_count / total_queries2 * 100) if total_queries2 > 0 else 0
                avg_overlap_percentage = (overlap_percentage1 + overlap_percentage2) / 2
                
                # Get metrics for both URLs
                url1_metrics = url_metrics[url_metrics['page'] == url1].iloc[0]
                url2_metrics = url_metrics[url_metrics['page'] == url2].iloc[0]
                
                overlap_data.append({
                    'url1': url1,
                    'url2': url2,
                    'url1_indexed_queries': int(url1_metrics['indexed_queries']),
                    'url1_clicks': int(url1_metrics['clicks']),
                    'url1_impressions': int(url1_metrics['impressions']),
                    'url2_indexed_queries': int(url2_metrics['indexed_queries']),
                    'url2_clicks': int(url2_metrics['clicks']),
                    'url2_impressions': int(url2_metrics['impressions']),
                    'keyword_overlap_count': int(overlap_count),
                    'keyword_overlap_percentage': round(avg_overlap_percentage, 2)
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
            # Use a simpler vectorizer to avoid issues
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create similarity dictionary
            similarity_scores = {}
            for i, url1 in enumerate(urls):
                for j, url2 in enumerate(urls):
                    if i < j:  # Only store unique pairs
                        similarity = max(0, similarity_matrix[i][j])  # Ensure non-negative
                        similarity_scores[(url1, url2)] = round(similarity * 100, 2)
            
            return similarity_scores
        except Exception as e:
            # Use a more robust simple similarity
            return self._calculate_robust_similarity(df)
    
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

    def _calculate_robust_similarity(self, df: pd.DataFrame) -> Dict:
        """Calculate robust similarity when TF-IDF fails"""
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        similarity_scores = {}
        
        urls = list(url_queries.keys())
        for i, url1 in enumerate(urls):
            for j, url2 in enumerate(urls):
                if i < j:
                    queries1 = url_queries[url1]
                    queries2 = url_queries[url2]
                    
                    # Calculate Jaccard similarity
                    intersection = len(queries1.intersection(queries2))
                    union = len(queries1.union(queries2))
                    
                    if union > 0:
                        jaccard_similarity = (intersection / union) * 100
                    else:
                        jaccard_similarity = 0
                    
                    # Also calculate overlap coefficient for additional context
                    min_queries = min(len(queries1), len(queries2))
                    overlap_coefficient = (intersection / min_queries * 100) if min_queries > 0 else 0
                    
                    # Use average of both metrics
                    final_similarity = (jaccard_similarity + overlap_coefficient) / 2
                    similarity_scores[(url1, url2)] = round(final_similarity, 2)
        
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
        """Generate consolidation recommendations in exact format requested"""
        recommendations = []
        
        for _, overlap in overlap_matrix.iterrows():
            url1, url2 = overlap['url1'], overlap['url2']
            
            # Get semantic similarity
            semantic_score = semantic_similarity.get(
                (url1, url2), 
                semantic_similarity.get((url2, url1), 0)
            )
            
            # Determine action and priority based on overlap and similarity
            overlap_pct = overlap['keyword_overlap_percentage']
            if overlap_pct >= 60 and semantic_score >= 50:
                action = 'Merge'
                priority = 'High'
            elif overlap_pct >= 40:
                action = 'Redirect'
                priority = 'Medium'
            elif overlap_pct >= 20:
                action = 'Optimize'
                priority = 'Medium'
            else:
                action = 'Monitor'
                priority = 'Low'
            
            recommendations.append({
                'primary_url': overlap['url1'],
                'primary_indexed_queries': overlap['url1_indexed_queries'],
                'primary_clicks': overlap['url1_clicks'],
                'primary_impressions': overlap['url1_impressions'],
                'secondary_url': overlap['url2'],
                'secondary_indexed_queries': overlap['url2_indexed_queries'],
                'secondary_clicks': overlap['url2_clicks'],
                'secondary_impressions': overlap['url2_impressions'],
                'semantic_similarity': semantic_score,
                'keyword_overlap_count': overlap['keyword_overlap_count'],
                'keyword_overlap_percentage': overlap['keyword_overlap_percentage'],
                'recommended_action': action,
                'priority': priority
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
                'priorities': {},
                'actions': {}
            }
        
        # Count actions
        actions = recommendations['recommended_action'].value_counts().to_dict()
        
        # Count priorities
        priorities = recommendations['priority'].value_counts().to_dict()
        
        return {
            'total_pairs': len(recommendations),
            'priorities': priorities,
            'actions': actions
        }
