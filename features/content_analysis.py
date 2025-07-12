"""
Content Analysis Module
Handles content gap analysis and merge recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

class ContentAnalyzer:
    """Analyze content for gap identification and merge recommendations"""
    
    def __init__(self):
        self.content_elements = [
            'headings', 'images', 'videos', 'tables', 
            'lists', 'code_snippets', 'internal_links', 
            'external_links', 'schema_markup'
        ]
    
    def generate_merge_recommendations(self, df: pd.DataFrame, query: str) -> Dict:
        """
        Generate content merge recommendations for a query
        
        Args:
            df: DataFrame with performance data
            query: Query to analyze
            
        Returns:
            Dictionary with merge recommendations
        """
        query_data = df[df['query'] == query]
        
        if query_data.empty:
            return {'error': 'No data found for query'}
        
        # Rank pages by performance
        pages_ranked = self._rank_pages_by_performance(query_data)
        
        # Identify primary page
        primary_page = pages_ranked.iloc[0] if not pages_ranked.empty else None
        
        if primary_page is None:
            return {'error': 'Unable to determine primary page'}
        
        # Generate recommendations
        recommendations = {
            'query': query,
            'primary_page': {
                'url': primary_page['page'],
                'current_clicks': primary_page['clicks'],
                'current_position': primary_page['avg_position']
            },
            'pages_to_redirect': [],
            'content_to_preserve': [],
            'estimated_impact': {},
            'implementation_priority': 'medium',
            'risks': []
        }
        
        # Analyze pages to redirect
        for _, page in pages_ranked.iloc[1:].iterrows():
            redirect_info = {
                'url': page['page'],
                'clicks_to_recover': page['clicks'],
                'unique_content_elements': self._identify_unique_content(page['page']),
                'redirect_type': '301'
            }
            recommendations['pages_to_redirect'].append(redirect_info)
        
        # Calculate estimated impact
        recommendations['estimated_impact'] = self._calculate_merge_impact(pages_ranked)
        
        # Assess risks
        recommendations['risks'] = self._assess_merge_risks(query_data)
        
        # Set priority
        recommendations['implementation_priority'] = self._determine_priority(
            recommendations['estimated_impact']['total_recovery_potential']
        )
        
        return recommendations
    
    def _rank_pages_by_performance(self, query_data: pd.DataFrame) -> pd.DataFrame:
        """Rank pages by performance metrics"""
        # Calculate composite performance score
        pages = query_data.groupby('page').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        
        pages.columns = ['page', 'clicks', 'impressions', 'avg_position']
        
        # Performance score: clicks weighted by position
        pages['performance_score'] = pages['clicks'] / (pages['avg_position'] + 1)
        
        # CTR for additional context
        pages['ctr'] = (pages['clicks'] / pages['impressions'] * 100) if pages['impressions'].sum() > 0 else 0
        
        return pages.sort_values('performance_score', ascending=False)
    
    def _identify_unique_content(self, page_url: str) -> List[str]:
        """Identify unique content elements on a page (simulated)"""
        # In production, this would analyze actual page content
        # For now, return simulated unique elements based on URL patterns
        
        unique_elements = []
        
        if '/blog/' in page_url:
            unique_elements.extend(['comments_section', 'author_bio', 'related_posts'])
        if '/guide/' in page_url:
            unique_elements.extend(['step_by_step_instructions', 'downloadable_pdf'])
        if '/product/' in page_url:
            unique_elements.extend(['pricing_table', 'customer_reviews', 'buy_button'])
        if any(year in page_url for year in ['2024', '2025', '2023']):
            unique_elements.append('year_specific_content')
        
        # Add some random elements for variety
        possible_elements = ['infographic', 'video_tutorial', 'case_study', 'faq_section']
        unique_elements.extend(np.random.choice(possible_elements, size=np.random.randint(0, 3), replace=False))
        
        return list(set(unique_elements))
    
    def _calculate_merge_impact(self, pages_ranked: pd.DataFrame) -> Dict:
        """Calculate estimated impact of content merge"""
        if len(pages_ranked) < 2:
            return {
                'total_recovery_potential': 0,
                'ctr_improvement': 0,
                'position_improvement': 0
            }
        
        primary_page = pages_ranked.iloc[0]
        secondary_pages = pages_ranked.iloc[1:]
        
        # Traffic recovery (70% of secondary page clicks)
        recovery_potential = int(secondary_pages['clicks'].sum() * 0.7)
        
        # CTR improvement (weighted average)
        total_impressions = pages_ranked['impressions'].sum()
        total_clicks = pages_ranked['clicks'].sum()
        current_ctr = (primary_page['clicks'] / primary_page['impressions'] * 100) if primary_page['impressions'] > 0 else 0
        potential_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        ctr_improvement = potential_ctr - current_ctr
        
        # Position improvement (estimated)
        position_improvement = max(0, primary_page['avg_position'] - 1.5)
        
        return {
            'total_recovery_potential': recovery_potential,
            'ctr_improvement': round(ctr_improvement, 2),
            'position_improvement': round(position_improvement, 1),
            'consolidation_strength': round((recovery_potential / total_clicks * 100) if total_clicks > 0 else 0, 1)
        }
    
    def _assess_merge_risks(self, query_data: pd.DataFrame) -> List[Dict]:
        """Assess risks associated with content merge"""
        risks = []
        
        # Check for significant position differences
        positions = query_data.groupby('page')['position'].mean()
        if positions.std() > 10:
            risks.append({
                'type': 'position_variance',
                'severity': 'medium',
                'description': 'Large position variance between pages may indicate different ranking factors'
            })
        
        # Check for seasonal patterns
        if self._detect_seasonal_patterns(query_data):
            risks.append({
                'type': 'seasonal_content',
                'severity': 'high',
                'description': 'Seasonal patterns detected - consolidation may impact time-sensitive rankings'
            })
        
        # Check for diverse URL patterns
        pages = query_data['page'].unique()
        if self._has_diverse_url_structure(pages):
            risks.append({
                'type': 'url_structure',
                'severity': 'low',
                'description': 'Diverse URL structures may serve different user segments'
            })
        
        return risks
    
    def _detect_seasonal_patterns(self, query_data: pd.DataFrame) -> bool:
        """Detect if content has seasonal patterns"""
        # Simplified check - in production would analyze time series data
        pages = query_data['page'].unique()
        seasonal_indicators = ['christmas', 'summer', 'winter', 'spring', 'fall', 'holiday', '2024', '2025']
        
        return any(indicator in page.lower() for page in pages for indicator in seasonal_indicators)
    
    def _has_diverse_url_structure(self, pages: List[str]) -> bool:
        """Check if pages have diverse URL structures"""
        structures = set()
        for page in pages:
            # Extract main directory
            parts = page.strip('/').split('/')
            if parts:
                structures.add(parts[0])
        
        return len(structures) > 1
    
    def _determine_priority(self, recovery_potential: int) -> str:
        """Determine implementation priority based on impact"""
        if recovery_potential > 1000:
            return 'high'
        elif recovery_potential > 500:
            return 'medium'
        else:
            return 'low'


class ContentGapAnalyzer:
    """Identify content gaps when consolidating pages"""
    
    def analyze_content_gaps(self, pages_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze content gaps between pages
        
        Args:
            pages_data: List of page data dictionaries
            
        Returns:
            DataFrame with gap analysis
        """
        gap_analysis = []
        
        for i, page1 in enumerate(pages_data):
            for page2 in pages_data[i+1:]:
                gaps = self._compare_page_content(page1, page2)
                gap_analysis.append(gaps)
        
        return pd.DataFrame(gap_analysis)
    
    def _compare_page_content(self, page1: Dict, page2: Dict) -> Dict:
        """Compare content between two pages"""
        # Simulated content comparison
        # In production, this would analyze actual page content
        
        page1_url = page1.get('url', 'page1')
        page2_url = page2.get('url', 'page2')
        
        # Simulate content analysis
        common_elements = np.random.randint(5, 15)
        unique_to_page1 = np.random.randint(2, 8)
        unique_to_page2 = np.random.randint(2, 8)
        
        overlap_percentage = (common_elements / (common_elements + unique_to_page1 + unique_to_page2)) * 100
        
        return {
            'page1': page1_url,
            'page2': page2_url,
            'common_elements': common_elements,
            'unique_to_page1': unique_to_page1,
            'unique_to_page2': unique_to_page2,
            'overlap_percentage': round(overlap_percentage, 1),
            'content_gap_severity': self._classify_gap_severity(overlap_percentage),
            'merge_difficulty': self._estimate_merge_difficulty(unique_to_page1, unique_to_page2)
        }
    
    def _classify_gap_severity(self, overlap_percentage: float) -> str:
        """Classify content gap severity"""
        if overlap_percentage > 70:
            return 'low'
        elif overlap_percentage > 40:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_merge_difficulty(self, unique1: int, unique2: int) -> str:
        """Estimate difficulty of merging content"""
        total_unique = unique1 + unique2
        
        if total_unique < 5:
            return 'easy'
        elif total_unique < 10:
            return 'moderate'
        else:
            return 'complex'
    
    def generate_preservation_plan(self, gap_analysis: pd.DataFrame) -> List[Dict]:
        """Generate content preservation plan based on gap analysis"""
        preservation_plan = []
        
        for _, gap in gap_analysis.iterrows():
            if gap['content_gap_severity'] in ['medium', 'high']:
                plan = {
                    'pages': f"{gap['page1']} + {gap['page2']}",
                    'action': 'careful_merge',
                    'preserve_from_page1': f"{gap['unique_to_page1']} unique elements",
                    'preserve_from_page2': f"{gap['unique_to_page2']} unique elements",
                    'estimated_time': f"{gap['unique_to_page1'] + gap['unique_to_page2']} hours",
                    'priority': 'high' if gap['content_gap_severity'] == 'high' else 'medium'
                }
                preservation_plan.append(plan)
        
        return preservation_plan
