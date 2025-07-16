"""
Enhanced Consolidation Analysis Module
Similarity-first approach using proven fast processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from collections import Counter


class EnhancedConsolidationAnalyzer:
    """Fast similarity-based consolidation analysis"""

    def __init__(self, max_urls: int = 1000, similarity_threshold: float = 0.85):
        self.max_urls = max_urls
        self.similarity_threshold = similarity_threshold

    def analyze_url_consolidation(self, df: pd.DataFrame) -> Dict:
        """Fast similarity-based analysis"""
        print(f"Processing {len(df)} rows...")
        
        # Step 1: Fast URL consolidation
        url_consolidated = self._fast_consolidate(df)
        print(f"Consolidated to {len(url_consolidated)} URLs")
        
        # Step 2: Limit to manageable number
        if len(url_consolidated) > self.max_urls:
            url_consolidated = url_consolidated.head(self.max_urls)
            print(f"Limited to top {self.max_urls} URLs")
        
        # Step 3: Ultra-fast similarity calculation
        similarity_matrix = self._calculate_similarity_fast(url_consolidated)
        print(f"Found {len(similarity_matrix)} similar pairs")
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            url_consolidated, similarity_matrix
        )
        
        return {
            'url_metrics': url_consolidated,
            'similarity_matrix': similarity_matrix,
            'consolidation_recommendations': recommendations,
            'summary': self._create_summary(recommendations)
        }

    def _fast_consolidate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast URL consolidation"""
        return df.groupby('page').agg({
            'query': lambda x: ' '.join(set(x)),  # Join queries for similarity
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index().rename(columns={
            'query': 'query_text',
            'clicks': 'total_clicks',
            'impressions': 'total_impressions'
        })

    def _calculate_similarity_fast(self, url_df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast similarity using simple text comparison"""
        # Create query text mapping
        url_queries = dict(zip(url_df['page'], url_df['query_text']))
        
        results = []
        urls = list(url_queries.keys())
        
        # Fast similarity calculation
        for i, url1 in enumerate(urls):
            text1 = url_queries[url1].lower()
            words1 = set(text1.split())
            
            for url2 in urls[i+1:]:
                text2 = url_queries[url2].lower()
                words2 = set(text2.split())
                
                # Fast Jaccard similarity
                intersection = words1 & words2
                union = words1 | words2
                
                if len(union) == 0:
                    continue
                
                similarity = len(intersection) / len(union)
                
                # Only include high similarity pairs
                if similarity >= self.similarity_threshold:
                    results.append({
                        'url1': url1,
                        'url2': url2,
                        'similarity_score': round(similarity, 3),
                        'common_words': len(intersection),
                        'total_words': len(union)
                    })
        
        return pd.DataFrame(results)

    def _generate_recommendations(self, url_df: pd.DataFrame, 
                                similarity_df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations from similarity"""
        if similarity_df.empty:
            return pd.DataFrame()
        
        # Create lookup
        url_lookup = url_df.set_index('page')[
            ['total_clicks', 'total_impressions', 'query_text']
        ].to_dict('index')
        
        recommendations = []
        
        for _, row in similarity_df.iterrows():
            url1, url2 = row['url1'], row['url2']
            
            # Get metrics
            clicks1 = url_lookup[url1]['total_clicks']
            clicks2 = url_lookup[url2]['total_clicks']
            
            # Count keywords (approximate)
            keywords1 = len(url_lookup[url1]['query_text'].split())
            keywords2 = len(url_lookup[url2]['query_text'].split())
            
            # Determine primary/secondary
            if clicks1 >= clicks2:
                primary, secondary = url1, url2
                primary_clicks, secondary_clicks = clicks1, clicks2
                primary_keywords, secondary_keywords = keywords1, keywords2
            else:
                primary, secondary = url2, url1
                primary_clicks, secondary_clicks = clicks2, clicks1
                primary_keywords, secondary_keywords = keywords2, keywords1
            
            # Simple consolidation logic
            similarity = row['similarity_score']
            if similarity >= 0.9:
                if secondary_keywords >= 100:
                    action = 'Optimize'
                elif secondary_clicks < primary_clicks * 0.2:
                    action = 'Redirect'
                else:
                    action = 'Merge'
            else:
                action = 'Internal Link'
            
            priority = 'High' if secondary_clicks > 1000 else \
                      'Medium' if secondary_clicks > 100 else 'Low'
            
            recommendations.append({
                'primary_page': primary,
                'primary_page_indexed_keywords': primary_keywords,
                'primary_page_clicks': primary_clicks,
                'secondary_page': secondary,
                'secondary_page_indexed_keywords': secondary_keywords,
                'secondary_page_clicks': secondary_clicks,
                'similarity_score': similarity,
                'number_keyword_overlaping': row['common_words'],
                'consolidation_type': action,
                'priority': priority
            })
        
        return pd.DataFrame(recommendations)

    def _create_summary(self, recommendations: pd.DataFrame) -> Dict:
        """Create summary"""
        if recommendations.empty:
            return {'total_recommendations': 0}
        
        return {
            'total_recommendations': len(recommendations),
            'optimize': len(recommendations[recommendations['consolidation_type'] == 'Optimize']),
            'merge': len(recommendations[recommendations['consolidation_type'] == 'Merge']),
            'redirect': len(recommendations[recommendations['consolidation_type'] == 'Redirect']),
            'internal_link': len(recommendations[recommendations['consolidation_type'] == 'Internal Link']),
            'high_priority': len(recommendations[recommendations['priority'] == 'High']),
            'medium_priority': len(recommendations[recommendations['priority'] == 'Medium']),
            'low_priority': len(recommendations[recommendations['priority'] == 'Low'])
        }
