"""
Optimized Consolidation Analyzer
Enhanced URL-level analysis with semantic similarity
"""

import pandas as pd
from typing import Dict
from features.simple_similarity_loader import SimpleSemanticSimilarityLoader


class OptimizedConsolidationAnalyzer:
    """Optimized URL consolidation analysis"""
    
    def __init__(self):
        self.similarity_loader = SimpleSemanticSimilarityLoader()
    
    def analyze_consolidation(self, df: pd.DataFrame, embeddings_df: pd.DataFrame) -> Dict:
        """
        Analyze URL consolidation opportunities using pre-calculated similarity
        
        Args:
            df: DataFrame with query-page performance data
            embeddings_df: DataFrame with URL similarity data
            
        Returns:
            Dictionary with URL consolidation analysis
        """
        import time
        start_time = time.time()
        
        # Load similarity data directly
        self.similarity_loader.load_similarity_data(embeddings_df)
        embeddings_used = True
        
        # Use all URLs from the dataset
        filtered_df = df
        print(f"Analyzing {len(filtered_df['page'].unique())} URLs...")
        
        # Calculate URL-level metrics
        url_metrics = self._calculate_url_metrics(filtered_df)
        
        # Generate consolidation recommendations
        recommendations = self._generate_consolidation_recommendations(
            filtered_df, url_metrics, embeddings_used
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
    
    def _generate_consolidation_recommendations(self, df: pd.DataFrame, 
                                              url_metrics: pd.DataFrame,
                                              embeddings_used: bool) -> pd.DataFrame:
        """Generate consolidation recommendations based on semantic similarity"""
        recommendations = []
        
        # Get significant URLs
        url_clicks = df.groupby('page')['clicks'].sum()
        significant_urls = url_clicks[url_clicks >= 5].index
        
        if len(significant_urls) > 200:
            significant_urls = url_clicks.nlargest(200).index
        
        # Create URL-query mapping
        url_queries = df[df['page'].isin(significant_urls)].groupby('page')['query'].apply(set).to_dict()
        urls = list(url_queries.keys())
        
        if len(urls) < 2:
            return pd.DataFrame()
        
        # Generate recommendations based on semantic similarity
        for i, url1 in enumerate(urls):
            for url2 in urls[i+1:]:
                similarity = self.similarity_loader.get_similarity_score(url1, url2)
                
                if similarity == 0.0:
                    continue
                
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
                
                # Traffic recovery estimate
                potential_recovery = int(secondary_metrics['total_clicks'] * 0.7)
                
                # Determine recommended action based on similarity
                recommended_action = self._determine_action(similarity, potential_recovery)
                
                # Determine priority
                priority = self._determine_priority(potential_recovery, similarity)
                
                recommendations.append({
                    'primary_url': primary_url,
                    'primary_indexed_queries': primary_metrics['num_queries'],
                    'primary_clicks': primary_metrics['total_clicks'],
                    'primary_impressions': primary_metrics['total_impressions'],
                    'secondary_url': secondary_url,
                    'secondary_indexed_queries': secondary_metrics['num_queries'],
                    'secondary_clicks': secondary_metrics['total_clicks'],
                    'secondary_impressions': secondary_metrics['total_impressions'],
                    'semantic_similarity': round(similarity, 2),
                    'recommended_action': recommended_action,
                    'priority': priority
                })
        
        return pd.DataFrame(recommendations)
    
    def _determine_action(self, similarity: float, potential_recovery: int) -> str:
        """Determine the recommended action based on similarity and metrics"""
        if similarity >= 0.8 and potential_recovery > 100:
            return 'Merge'
        elif similarity >= 0.6 and potential_recovery > 50:
            return 'Redirect'
        elif similarity >= 0.4:
            return 'Optimize'
        elif similarity >= 0.2:
            return 'Internal Link'
        else:
            return 'Monitor'
    
    def _determine_priority(self, potential_recovery: int, similarity: float) -> str:
        """Determine the priority based on potential recovery and similarity"""
        if potential_recovery > 500 and similarity > 0.7:
            return 'High'
        elif potential_recovery > 100 and similarity > 0.5:
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
