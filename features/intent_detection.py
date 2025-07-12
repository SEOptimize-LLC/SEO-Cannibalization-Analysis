"""
Intent Mismatch Detection Module
Identifies when pages ranking for the same query may have different user intents
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import ANALYSIS_DEFAULTS

class IntentDetector:
    """Detect intent mismatches between pages ranking for the same query"""
    
    def __init__(self, threshold: float = ANALYSIS_DEFAULTS['intent_mismatch_threshold']):
        self.threshold = threshold
        self.intent_signals = {
            'informational': ['what', 'how', 'why', 'guide', 'tutorial', 'learn'],
            'transactional': ['buy', 'price', 'deal', 'discount', 'shop', 'order'],
            'navigational': ['login', 'sign in', 'official', 'website'],
            'commercial': ['best', 'top', 'review', 'compare', 'vs']
        }
    
    def analyze_query_intents(self, df: pd.DataFrame, query: str) -> Dict:
        """
        Analyze intent mismatch for a specific query
        
        Args:
            df: DataFrame with search console data
            query: Query to analyze
            
        Returns:
            Dictionary with intent analysis results
        """
        query_data = df[df['query'] == query]
        
        if len(query_data) < 2:
            return {
                'query': query,
                'intent_mismatch': False,
                'score': 0,
                'details': 'Not enough pages to analyze'
            }
        
        # Analyze various signals
        position_analysis = self._analyze_position_variance(query_data)
        click_analysis = self._analyze_click_patterns(query_data)
        url_analysis = self._analyze_url_patterns(query_data)
        query_intent = self._classify_query_intent(query)
        
        # Calculate composite score
        intent_score = (
            position_analysis['normalized_variance'] * 0.3 +
            click_analysis['distribution_score'] * 0.3 +
            url_analysis['diversity_score'] * 0.4
        )
        
        return {
            'query': query,
            'intent_mismatch': intent_score > self.threshold,
            'score': round(intent_score, 2),
            'primary_intent': query_intent,
            'position_variance': position_analysis,
            'click_patterns': click_analysis,
            'url_patterns': url_analysis,
            'recommendation': self._generate_recommendation(intent_score, query_intent)
        }
    
    def _analyze_position_variance(self, query_data: pd.DataFrame) -> Dict:
        """Analyze position variance as intent signal"""
        positions = query_data.groupby('page')['position'].mean()
        
        variance = positions.var()
        std_dev = positions.std()
        position_gap = positions.max() - positions.min()
        
        # Normalize variance (0-1 scale)
        normalized_variance = min(variance / 100, 1) if variance else 0
        
        return {
            'variance': round(variance, 2),
            'std_dev': round(std_dev, 2),
            'position_gap': round(position_gap, 2),
            'normalized_variance': round(normalized_variance, 2),
            'high_variance': normalized_variance > 0.5
        }
    
    def _analyze_click_patterns(self, query_data: pd.DataFrame) -> Dict:
        """Analyze click distribution patterns"""
        page_clicks = query_data.groupby('page')['clicks'].sum()
        
        # Calculate distribution metrics
        total_clicks = page_clicks.sum()
        if total_clicks == 0:
            return {
                'distribution_score': 0,
                'top_page_dominance': 0,
                'click_entropy': 0
            }
        
        click_shares = page_clicks / total_clicks
        
        # Calculate entropy (higher = more distributed)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in click_shares)
        max_entropy = np.log2(len(click_shares)) if len(click_shares) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Top page dominance
        top_page_share = click_shares.max()
        
        return {
            'distribution_score': round(normalized_entropy, 2),
            'top_page_dominance': round(top_page_share, 2),
            'click_entropy': round(entropy, 2),
            'distributed_clicks': normalized_entropy > 0.5
        }
    
    def _analyze_url_patterns(self, query_data: pd.DataFrame) -> Dict:
        """Analyze URL patterns for intent signals"""
        pages = query_data['page'].unique()
        
        # Extract URL components
        url_patterns = {
            'blog': sum(1 for p in pages if '/blog/' in p.lower()),
            'product': sum(1 for p in pages if any(x in p.lower() for x in ['/product/', '/shop/', '/buy/'])),
            'category': sum(1 for p in pages if '/category/' in p.lower()),
            'guide': sum(1 for p in pages if any(x in p.lower() for x in ['/guide/', '/how-to/', '/tutorial/'])),
            'comparison': sum(1 for p in pages if any(x in p.lower() for x in ['/vs/', '/compare/', '/versus/']))
        }
        
        # Calculate diversity score
        pattern_counts = [v for v in url_patterns.values() if v > 0]
        diversity_score = len(pattern_counts) / len(url_patterns) if pattern_counts else 0
        
        return {
            'diversity_score': round(diversity_score, 2),
            'url_patterns': url_patterns,
            'pattern_count': len(pattern_counts),
            'diverse_urls': diversity_score > 0.3
        }
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the primary intent of a query"""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, signals in self.intent_signals.items():
            score = sum(1 for signal in signals if signal in query_lower)
            intent_scores[intent] = score
        
        # Return intent with highest score, or 'mixed' if unclear
        max_score = max(intent_scores.values())
        if max_score == 0:
            return 'unclear'
        
        top_intents = [k for k, v in intent_scores.items() if v == max_score]
        return top_intents[0] if len(top_intents) == 1 else 'mixed'
    
    def _generate_recommendation(self, score: float, intent: str) -> str:
        """Generate recommendation based on intent analysis"""
        if score < 0.3:
            return "Low intent mismatch - safe to consolidate"
        elif score < 0.6:
            return f"Moderate intent mismatch - review content alignment for {intent} intent"
        else:
            return "High intent mismatch - consider keeping pages separate or creating intent-specific content"
    
    def batch_analyze(self, df: pd.DataFrame, queries: List[str]) -> pd.DataFrame:
        """Analyze multiple queries for intent mismatches"""
        results = []
        
        for query in queries:
            analysis = self.analyze_query_intents(df, query)
            results.append({
                'query': query,
                'intent_mismatch_score': analysis['score'],
                'likely_mismatch': analysis['intent_mismatch'],
                'primary_intent': analysis['primary_intent'],
                'recommendation': analysis['recommendation']
            })
        
        return pd.DataFrame(results)


class ContentIntentAnalyzer:
    """Analyze content-level intent alignment"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def analyze_content_alignment(self, page_contents: Dict[str, str]) -> Dict:
        """
        Analyze content alignment between pages
        
        Args:
            page_contents: Dictionary mapping page URLs to content
            
        Returns:
            Dictionary with content alignment analysis
        """
        if len(page_contents) < 2:
            return {
                'alignment_score': 1.0,
                'similar_pages': [],
                'different_pages': []
            }
        
        # Vectorize content
        pages = list(page_contents.keys())
        contents = list(page_contents.values())
        
        try:
            content_vectors = self.vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(content_vectors)
            
            # Analyze similarities
            similar_pairs = []
            different_pairs = []
            
            for i in range(len(pages)):
                for j in range(i + 1, len(pages)):
                    similarity = similarity_matrix[i][j]
                    pair = {
                        'page1': pages[i],
                        'page2': pages[j],
                        'similarity': round(similarity, 2)
                    }
                    
                    if similarity > 0.7:
                        similar_pairs.append(pair)
                    elif similarity < 0.3:
                        different_pairs.append(pair)
            
            # Calculate overall alignment
            avg_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()
            
            return {
                'alignment_score': round(avg_similarity, 2),
                'similar_pages': similar_pairs,
                'different_pages': different_pairs,
                'recommendation': self._get_alignment_recommendation(avg_similarity)
            }
            
        except Exception as e:
            return {
                'alignment_score': 0,
                'error': str(e),
                'recommendation': 'Unable to analyze content alignment'
            }
    
    def _get_alignment_recommendation(self, score: float) -> str:
        """Generate recommendation based on content alignment"""
        if score > 0.7:
            return "High content overlap - strong candidate for consolidation"
        elif score > 0.4:
            return "Moderate overlap - review unique content before consolidating"
        else:
            return "Low content overlap - pages serve different purposes"
