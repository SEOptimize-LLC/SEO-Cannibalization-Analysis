"""
Enhanced Consolidation Analysis Module
Provides URL-level analysis with keyword overlap, semantic similarity,
and comprehensive metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class EnhancedConsolidationAnalyzer:
    """Enhanced consolidation analysis with URL-level metrics"""

    def __init__(self, use_semantic_similarity: bool = False):
        self.use_semantic_similarity = use_semantic_similarity
        self.similarity_cache = {}

    def analyze_url_consolidation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze URL consolidation opportunities with enhanced metrics

        Args:
            df: DataFrame with query-page performance data

        Returns:
            Dictionary with URL-level consolidation analysis
        """
        # Calculate URL-level metrics
        url_metrics = self._calculate_url_metrics(df)

        # Calculate keyword overlap between URL pairs
        url_overlap_matrix = self._calculate_keyword_overlap(df)

        # Calculate semantic similarity (if enabled)
        semantic_similarity = {}
        if self.use_semantic_similarity:
            semantic_similarity = self._calculate_semantic_similarity(df)

        # Generate consolidation recommendations
        recommendations = self._generate_enhanced_recommendations(
            df, url_metrics, url_overlap_matrix, semantic_similarity
        )

        # Create summary statistics
        summary = self._create_consolidation_summary(recommendations)

        return {
            'url_metrics': url_metrics,
            'keyword_overlap': url_overlap_matrix,
            'semantic_similarity': semantic_similarity,
            'consolidation_recommendations': recommendations,
            'summary': summary
        }

    def _calculate_url_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive metrics for each URL"""
        url_metrics = df.groupby('page').agg({
            'query': lambda x: list(x.unique()),
            'clicks': 'sum',
            'impressions': 'sum',
            'position': ['mean', 'min', 'max']
        }).reset_index()

        url_metrics.columns = [
            'page', 'queries', 'total_clicks', 'total_impressions',
            'avg_position', 'best_position', 'worst_position'
        ]

        # Calculate additional metrics
        url_metrics['num_queries'] = url_metrics['queries'].apply(len)
        url_metrics['ctr'] = (
            url_metrics['total_clicks'] / url_metrics['total_impressions'] * 100
        ).round(2)
        url_metrics['avg_position'] = url_metrics['avg_position'].round(2)

        return url_metrics

    def _calculate_keyword_overlap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate keyword overlap between URL pairs"""
        # Create URL-query mapping
        url_queries = df.groupby('page')['query'].apply(set).to_dict()

        # Calculate overlap for all URL pairs
        overlap_data = []
        urls = list(url_queries.keys())

        for i, url1 in enumerate(urls):
            for url2 in urls[i+1:]:
                queries1 = url_queries[url1]
                queries2 = url_queries[url2]

                # Calculate overlap metrics
                intersection = queries1.intersection(queries2)
                union = queries1.union(queries2)

                overlap_count = len(intersection)
                total_queries1 = len(queries1)
                total_queries2 = len(queries2)

                overlap_percentage1 = (
                    overlap_count / total_queries1 * 100
                ) if total_queries1 > 0 else 0
                overlap_percentage2 = (
                    overlap_count / total_queries2 * 100
                ) if total_queries2 > 0 else 0
                jaccard_similarity = (
                    len(intersection) / len(union) * 100
                ) if len(union) > 0 else 0

                # Calculate combined performance for overlapping queries
                overlap_queries_df = df[df['query'].isin(intersection)]
                url1_overlap = overlap_queries_df[overlap_queries_df['page'] == url1]
                url2_overlap = overlap_queries_df[overlap_queries_df['page'] == url2]

                combined_clicks = (
                    url1_overlap['clicks'].sum() + url2_overlap['clicks'].sum()
                )
                combined_impressions = (
                    url1_overlap['impressions'].sum() +
                    url2_overlap['impressions'].sum()
                )

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
                    'combined_overlap_impressions': combined_impressions,
                    'overlap_strength': self._calculate_overlap_strength(
                        overlap_percentage1, overlap_percentage2, combined_clicks
                    )
                })

        return pd.DataFrame(overlap_data)

    def _calculate_overlap_strength(self, overlap1: float, overlap2: float,
                                  combined_clicks: int) -> str:
        """Calculate overlap strength based on percentage and traffic"""
        avg_overlap = (overlap1 + overlap2) / 2

        if avg_overlap >= 70 and combined_clicks > 100:
            return 'high'
        elif avg_overlap >= 40 and combined_clicks > 50:
            return 'medium'
        elif avg_overlap >= 20:
            return 'low'
        else:
            return 'minimal'

    def _calculate_semantic_similarity(self, df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """Calculate semantic similarity between URL content (simplified)"""
        url_queries = df.groupby('page')['query'].apply(list).to_dict()

        semantic_scores = {}
        urls = list(url_queries.keys())

        for i, url1 in enumerate(urls):
            for url2 in urls[i+1:]:
                queries1 = ' '.join(url_queries[url1]).lower()
                queries2 = ' '.join(url_queries[url2]).lower()

                # Simple similarity based on query overlap
                words1 = set(queries1.split())
                words2 = set(queries2.split())

                intersection = words1.intersection(words2)
                union = words1.union(words2)

                similarity = len(intersection) / len(union) if union else 0
                semantic_scores[(url1, url2)] = round(similarity * 100, 2)

        return semantic_scores

    def _generate_enhanced_recommendations(self, df: pd.DataFrame,
                                         url_metrics: pd.DataFrame,
                                         overlap_matrix: pd.DataFrame,
                                         semantic_similarity: Dict) -> pd.DataFrame:
        """Generate consolidation recommendations with exact output format"""
        recommendations = []

        for _, overlap in overlap_matrix.iterrows():
            url1, url2 = overlap['url1'], overlap['url2']

            # Get URL metrics
            url1_metrics = url_metrics[url_metrics['page'] == url1].iloc[0]
            url2_metrics = url_metrics[url_metrics['page'] == url2].iloc[0]

            # Determine primary URL (higher clicks) and secondary URL (lower clicks)
            if url1_metrics['total_clicks'] >= url2_metrics['total_clicks']:
                primary_url, secondary_url = url1, url2
                primary_metrics, secondary_metrics = url1_metrics, url2_metrics
            else:
                primary_url, secondary_url = url2, url1
                primary_metrics, secondary_metrics = url2_metrics, url1_metrics

            # Calculate similarity score (using jaccard similarity)
            similarity_score = overlap['jaccard_similarity']

            # Determine consolidation type based on your exact criteria
            consolidation_type = self._determine_consolidation_type(
                similarity_score,
                secondary_metrics['total_clicks'],
                secondary_metrics['num_queries'],
                primary_metrics['total_clicks'],
                primary_metrics['num_queries']
            )

            # Determine priority based on traffic potential
            priority = self._determine_priority_based_on_traffic(
                secondary_metrics['total_clicks']
            )

            # Build recommendation with exact output format
            recommendations.append({
                'primary_page': primary_url,
                'primary_page_indexed_keywords': primary_metrics['num_queries'],
                'primary_page_clicks': primary_metrics['total_clicks'],
                'secondary_page': secondary_url,
                'secondary_page_indexed_keywords': secondary_metrics['num_queries'],
                'secondary_page_clicks': secondary_metrics['total_clicks'],
                'similarity_score': similarity_score,
                'number_keyword_overlaping': overlap['overlap_count'],
                'consolidation_type': consolidation_type,
                'priority': priority
            })

        return pd.DataFrame(recommendations)

    def _determine_consolidation_type(self, similarity_score: float,
                                    secondary_clicks: int, secondary_keywords: int,
                                    primary_clicks: int, primary_keywords: int) -> str:
        """Determine consolidation type based on exact criteria"""
        if similarity_score >= 90:
            # Check criteria in order
            if secondary_keywords >= 100:
                return 'Optimize'
            elif abs(secondary_clicks - primary_clicks) < (primary_clicks * 0.2) and \
                 abs(secondary_keywords - primary_keywords) < (primary_keywords * 0.2):
                return 'Merge'
            elif secondary_clicks == 0 and secondary_keywords == 0:
                return 'Remove'
            elif secondary_clicks < (primary_clicks + secondary_clicks) * 0.2:
                return 'Redirect'
            else:
                return 'Redirect'  # Default for 90%+ similarity
        else:
            return 'Internal Link'

    def _determine_priority_based_on_traffic(self, secondary_clicks: int) -> str:
        """Determine priority based on secondary URL traffic"""
        if secondary_clicks > 1000:
            return 'High'
        elif secondary_clicks > 100:
            return 'Medium'
        else:
            return 'Low'

    def _create_consolidation_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary statistics for consolidation recommendations"""
        if recommendations.empty:
            return {
                'total_recommendations': 0,
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'total_potential_recovery': 0,
                'average_confidence': 0
            }

        return {
            'total_recommendations': len(recommendations),
            'high_priority': len(
                recommendations[recommendations['priority'] == 'high']
            ),
            'medium_priority': len(
                recommendations[recommendations['priority'] == 'medium']
            ),
            'low_priority': len(
                recommendations[recommendations['priority'] == 'low']
            ),
            'total_potential_recovery': recommendations['potential_traffic_recovery'].sum(),
            'average_confidence': recommendations['confidence_score'].mean(),
            'merge_and_redirect': len(
                recommendations[recommendations['recommendation_type'] == 'merge_and_redirect']
            ),
            'redirect_secondary': len(
                recommendations[recommendations['recommendation_type'] == 'redirect_secondary']
            ),
            'evaluate_content_merge': len(
                recommendations[recommendations['recommendation_type'] == 'evaluate_content_merge']
            ),
            'monitor_and_optimize': len(
                recommendations[recommendations['recommendation_type'] == 'monitor_and_optimize']
            )
        }
