"""
Core Cannibalization Analysis Logic
Adapted from the original helpers.py with enhancements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config.settings import ANALYSIS_DEFAULTS

class CannibalizationAnalyzer:
    """Main analyzer for SEO cannibalization detection"""
    
    def __init__(self, 
                 click_threshold: float = ANALYSIS_DEFAULTS['click_threshold'],
                 min_clicks: int = ANALYSIS_DEFAULTS['min_clicks'],
                 brand_variants: List[str] = None):
        self.click_threshold = click_threshold
        self.min_clicks = min_clicks
        self.brand_variants = brand_variants or []
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete cannibalization analysis
        
        Returns:
            Dictionary containing analysis results
        """
        # Remove brand queries
        df_filtered = self._remove_brand_queries(df)
        
        # Calculate metrics
        query_page_metrics = self._calculate_query_page_metrics(df_filtered)
        
        # Filter queries
        filtered_queries = self._filter_queries_by_criteria(query_page_metrics)
        
        # Merge and aggregate
        working_df = self._merge_and_aggregate(query_page_metrics, filtered_queries)
        
        # Calculate percentages
        working_df = self._calculate_click_percentages(working_df)
        
        # Filter by threshold
        working_df = self._filter_by_click_percentage(working_df)
        
        # Merge with page data
        working_df = self._merge_with_page_clicks(working_df, df)
        
        # Define opportunities
        final_df = self._define_opportunity_levels(working_df)
        
        # Sort and finalize
        final_df = self._sort_and_finalize(final_df)
        
        # Generate summary statistics
        summary = self._generate_summary_stats(final_df, df)
        
        # Identify immediate opportunities
        immediate_opps = self._identify_immediate_opportunities(final_df)
        
        return {
            'all_opportunities': final_df,
            'immediate_opportunities': immediate_opps,
            'summary': summary,
            'risk_qa_data': self._create_qa_dataframe(df, final_df)
        }
    
    def _remove_brand_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove queries containing brand variants"""
        if not self.brand_variants:
            return df
        
        pattern = '|'.join(self.brand_variants)
        return df[~df['query'].str.contains(pattern, case=False, na=False)]
    
    def _calculate_query_page_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for each query-page combination"""
        return df.groupby(['query', 'page']).agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index().rename(columns={'position': 'avg_position'})
    
    def _filter_queries_by_criteria(self, query_page_counts: pd.DataFrame) -> pd.DataFrame:
        """Filter queries based on click and page criteria"""
        query_counts = query_page_counts.groupby('query').agg({
            'page': 'nunique',
            'clicks': 'sum'
        }).reset_index()
        
        return query_counts[
            (query_counts['page'] >= ANALYSIS_DEFAULTS['min_pages']) & 
            (query_counts['clicks'] >= self.min_clicks)
        ]
    
    def _merge_and_aggregate(self, query_page_counts: pd.DataFrame, 
                           query_counts: pd.DataFrame) -> pd.DataFrame:
        """Merge metrics and aggregate"""
        df = query_page_counts.merge(query_counts[['query']], on='query', how='inner')
        return df.groupby(['page', 'query']).agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'avg_position': 'mean'
        }).reset_index()
    
    def _calculate_click_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate click percentages vs query total"""
        df['clicks_pct_vs_query'] = df.groupby('query')['clicks'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        return df
    
    def _filter_by_click_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter queries with multiple pages above threshold"""
        queries_to_keep = df[df['clicks_pct_vs_query'] >= self.click_threshold]\
            .groupby('query')\
            .filter(lambda x: len(x) >= 2)['query']\
            .unique()
        
        return df[df['query'].isin(queries_to_keep)]
    
    def _merge_with_page_clicks(self, working_df: pd.DataFrame, 
                               initial_df: pd.DataFrame) -> pd.DataFrame:
        """Merge with page-level click metrics"""
        page_clicks = initial_df.groupby('page').agg({'clicks': 'sum'}).reset_index()
        working_df = working_df.merge(page_clicks, on='page', how='inner')
        
        working_df['clicks_pct_vs_page'] = (
            working_df['clicks_x'] / working_df['clicks_y']
        ).fillna(0)
        
        return working_df.rename(columns={
            'clicks_x': 'clicks_query',
            'clicks_y': 'clicks_page'
        })
    
    def _define_opportunity_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define opportunity levels based on thresholds"""
        conditions = [
            (df['clicks_pct_vs_query'] >= self.click_threshold) & 
            (df['clicks_pct_vs_page'] >= self.click_threshold),
            (df['clicks_pct_vs_query'] < self.click_threshold) | 
            (df['clicks_pct_vs_page'] < self.click_threshold)
        ]
        
        choices = [
            'Potential Opportunity',
            'Risk - Low percentage of query or page clicks'
        ]
        
        df['comment'] = np.select(conditions, choices, default='Needs Review')
        
        # Add opportunity score
        df['opportunity_score'] = (
            df['clicks_pct_vs_query'] * 0.5 + 
            df['clicks_pct_vs_page'] * 0.5
        ) * 10  # Scale to 0-10
        
        return df
    
    def _sort_and_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort and format final output"""
        df = df.sort_values(
            ['query', 'clicks_pct_vs_query', 'clicks_pct_vs_page'],
            ascending=[True, False, False]
        )
        
        # Select and order columns
        column_order = [
            'query', 'page', 'impressions', 'avg_position',
            'clicks_query', 'clicks_pct_vs_query',
            'clicks_page', 'clicks_pct_vs_page',
            'opportunity_score', 'comment'
        ]
        
        df = df[column_order]
        
        # Round numeric columns
        numeric_cols = ['avg_position', 'clicks_pct_vs_query', 
                       'clicks_pct_vs_page', 'opportunity_score']
        df[numeric_cols] = df[numeric_cols].round(2)
        
        return df
    
    def _generate_summary_stats(self, final_df: pd.DataFrame, 
                               original_df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        total_queries = final_df['query'].nunique()
        total_pages = final_df['page'].nunique()
        
        opportunities = final_df[final_df['comment'] == 'Potential Opportunity']
        risks = final_df[final_df['comment'].str.contains('Risk', na=False)]
        
        # Calculate potential traffic recovery
        potential_recovery = self._calculate_traffic_recovery(opportunities)
        
        return {
            'total_queries_analyzed': len(original_df['query'].unique()),
            'cannibalization_issues': total_queries,
            'pages_affected': total_pages,
            'immediate_opportunities': len(opportunities),
            'risk_queries': len(risks),
            'potential_traffic_recovery': potential_recovery,
            'average_opportunity_score': opportunities['opportunity_score'].mean()
        }
    
    def _calculate_traffic_recovery(self, opportunities_df: pd.DataFrame) -> int:
        """Calculate potential traffic recovery from consolidation"""
        if opportunities_df.empty:
            return 0
        
        # Group by query and calculate potential recovery
        recovery = 0
        for query in opportunities_df['query'].unique():
            query_data = opportunities_df[opportunities_df['query'] == query]
            if len(query_data) > 1:
                total_clicks = query_data['clicks_query'].sum()
                max_clicks = query_data['clicks_query'].max()
                # Estimate 70% recovery of lost clicks
                recovery += (total_clicks - max_clicks) * ANALYSIS_DEFAULTS['traffic_recovery_estimate']
        
        return int(recovery)
    
    def _identify_immediate_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify queries with 2+ pages marked as opportunities"""
        return df[df['comment'] == 'Potential Opportunity']\
            .groupby('query')\
            .filter(lambda x: len(x) >= 2)
    
    def _create_qa_dataframe(self, initial_df: pd.DataFrame, 
                            final_df: pd.DataFrame) -> pd.DataFrame:
        """Create QA dataframe for risk assessment"""
        pages_in_analysis = final_df['page'].unique()
        return initial_df[initial_df['page'].isin(pages_in_analysis)]


class RedirectMapper:
    """Generate redirect maps for consolidation"""
    
    @staticmethod
    def build_redirect_map(df: pd.DataFrame, 
                          click_threshold: float = 0.5) -> pd.DataFrame:
        """
        Build redirect map based on click performance
        
        Args:
            df: DataFrame with cannibalization data
            click_threshold: Minimum click percentage to consider
            
        Returns:
            DataFrame with redirect recommendations
        """
        redirect_data = []
        
        for query in df['query'].unique():
            group = df[df['query'] == query]
            
            # Filter by click threshold
            filtered = group[group['clicks_pct_vs_page'] > click_threshold]
            
            if len(filtered) >= 2:
                # Sort by clicks to find target page
                filtered_sorted = filtered.sort_values('clicks_query', ascending=False)
                url_to = filtered_sorted.iloc[0]['page']
                
                # Create redirect entries for other pages
                for _, row in filtered_sorted.iloc[1:].iterrows():
                    redirect_data.append({
                        'query': query,
                        'url_from': row['page'],
                        'url_to': url_to,
                        'url_from_clicks': row['clicks_query'],
                        'url_to_clicks': filtered_sorted.iloc[0]['clicks_query'],
                        'confidence_score': row['opportunity_score']
                    })
        
        return pd.DataFrame(redirect_data)
