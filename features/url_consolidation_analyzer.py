"""
URL Consolidation Analyzer
Enhanced URL-level analysis with keyword overlap and semantic similarity
"""

import pandas as pd
from typing import Dict
from features.simple_similarity_loader import SimpleSemanticSimilarityLoader


class URLConsolidationAnalyzer:
    """Enhanced URL consolidation analysis"""
    
    def __init__(self):
        self.similarity_loader = SimpleSemanticSimilarityLoader()
    
    def analyze_url_consolidation(self, df: pd.DataFrame, embeddings_df: pd.DataFrame = None) -> Dict:
        """
        Analyze URL consolidation opportunities with performance optimization
        
        Args:
            df: DataFrame with query-page performance data
            embeddings_df: Optional embeddings DataFrame for semantic similarity
            
        Returns:
            Dictionary with URL consolidation analysis
        """
        import time
        start_time = time.time()
        
        # Load semantic similarity data if embeddings provided
        embeddings_used = False
        if embeddings_df is not None:
            try:
                # Save embeddings to temporary file for similarity loader
                temp_file = "temp_embeddings.csv"
                embeddings_df.to_csv(temp_file, index=False)
                self.similarity_loader.load_similarity_data(temp_file)
                embeddings_used = True
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Could not load embeddings: {e}")
        
        # Use all URLs from the dataset - no artificial limit
        filtered_df = df
        print(f"Analyzing {len(filtered_df['page'].unique())} URLs...")
        
        # Calculate URL-level metrics
        url_metrics = self._calculate_url_metrics(filtered_df)
        
        # Calculate keyword overlap between URLs - optimized version
        url_overlap_matrix = self._calculate_keyword_overlap_optimized(filtered_df)
        
        # Generate consolidation recommendations
        recommendations = self._generate_consolidation_recommendations(
            filtered_df, url_metrics, url_overlap_matrix, embeddings_used
        )
        
        # Create summary
        summary = self._create_summary(recommendations)
        
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return {
            'recommendations': recommendations,
            'summary': summary,
            'embeddings_used': embeddings_used,
            'url_metrics': url_metrics,
            'overlap_matrix': url_overlap_matrix,
            'urls_analyzed': len(url_metrics)
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
        return self._calculate_keyword_overlap_optimized(df)
    
    def _calculate_keyword_overlap_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized keyword overlap calculation using vectorized operations"""
        # Create URL-query mapping efficiently
        url_queries = df.groupby('page')['query'].apply(set).to_dict()
        urls = list(url_queries.keys())
        
        if len(urls) < 2:
            return pd.DataFrame()
        
        overlap_data = []
        
        # Use efficient iteration with early termination
        for i, url1 in enumerate(urls):
            queries1 = url_queries[url1]
            if not queries1:  # Skip empty query sets
                continue
                
            for url2 in urls[i+1:]:
                queries2 = url_queries[url2]
                if not queries2:  # Skip empty query sets
                    continue
                
                # Calculate overlap efficiently
                intersection = queries1.intersection(queries2)
                if not intersection:  # Skip no-overlap pairs
                    continue
                
                overlap_count = len(intersection)
                total_queries1 = len(queries1)
                total_queries2 = len(queries2)
                
                # Calculate percentages
                overlap_percentage1 = (overlap_count / total_queries1 * 100) if total_queries1 > 0 else 0
                overlap_percentage2 = (overlap_count / total_queries2 * 100) if total_queries2 > 0 else 0
                
                overlap_queries = list(intersection)
                
                overlap_data.append({
                    'url1': url1,
                    'url2': url2,
                    'overlap_queries': overlap_queries,
                    'overlap_count': overlap_count,
                    'url1_total_queries': total_queries1,
                    'url2_total_queries': total_queries2,
                    'overlap_percentage_url1': round(overlap_percentage1, 2),
                    'overlap_percentage_url2': round(overlap_percentage2, 2)
                })
        
        return pd.DataFrame(overlap_data)
    
    def _generate_consolidation_recommendations(self, df: pd.DataFrame, 
                                              url_metrics: pd.DataFrame,
                                              overlap_matrix: pd.DataFrame,
                                              embeddings_used: bool) -> pd.DataFrame:
        """Generate consolidation recommendations based on URL analysis"""
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
            else:
                primary_url, secondary_url = url2, url1
                primary_metrics, secondary_metrics = url2_metrics, url1_metrics
            
            # Get semantic similarity score
            semantic_similarity = self.similarity_loader.get_similarity_score(primary_url, secondary_url)
            
            # Traffic recovery estimate (70% of secondary URL clicks)
            potential_recovery = int(secondary_metrics['total_clicks'] * 0.7)
            
            # Determine recommended action
            recommended_action = self._determine_action(
                overlap, semantic_similarity, potential_recovery
            )
            
            # Determine priority
            priority = self._determine_priority(potential_recovery, overlap['overlap_count'])
            
            recommendations.append({
                'primary_url': primary_url,
                'primary_indexed_queries': primary_metrics['num_queries'],
                'primary_clicks': primary_metrics['total_clicks'],
                'primary_impressions': primary_metrics['total_impressions'],
                'secondary_url': secondary_url,
                'secondary_indexed_queries': secondary_metrics['num_queries'],
                'secondary_clicks': secondary_metrics['total_clicks'],
                'secondary_impressions': secondary_metrics['total_impressions'],
                'keyword_overlap_count': overlap['overlap_count'],
                'keyword_overlap_percentage': max(overlap['overlap_percentage_url1'], overlap['overlap_percentage_url2']),
                'semantic_similarity': round(semantic_similarity, 2),
                'recommended_action': recommended_action,
                'priority': priority
            })
        
        return pd.DataFrame(recommendations)
    
    def _determine_action(self, overlap: pd.Series, semantic_similarity: float, potential_recovery: int) -> str:
        """Determine the recommended action based on overlap and metrics"""
        overlap_percentage = max(overlap['overlap_percentage_url1'], overlap['overlap_percentage_url2'])
        
        if overlap_percentage >= 60 and potential_recovery > 100:
            return 'Merge'
        elif overlap_percentage >= 40 and potential_recovery > 50:
            return 'Redirect'
        elif overlap_percentage >= 20 and semantic_similarity >= 0.7:
            return 'Optimize'
        elif overlap_percentage >= 10:
            return 'Internal Link'
        elif overlap_percentage >= 5:
            return 'Monitor'
        else:
            return 'False Positive'
    
    def _determine_priority(self, potential_recovery: int, overlap_count: int) -> str:
        """Determine the priority based on potential recovery and overlap"""
        if potential_recovery > 500 and overlap_count > 10:
            return 'High'
        elif potential_recovery > 100 and overlap_count > 5:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics for recommendations"""
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
