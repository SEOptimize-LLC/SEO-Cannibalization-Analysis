"""
URL filtering module - removes unwanted URLs based on specific rules
"""

import re


class URLFilter:
    """Filters out unwanted URLs based on specific rules"""
    
    @staticmethod
    def should_filter_url(url: str) -> bool:
        """Determine if URL should be filtered out"""
        url_lower = url.lower()
        
        # Filter patterns
        patterns = [
            r'[?=#&]',  # Parameters
            r'/privacy', r'/terms', r'/legal', r'/shipping',  # Legal pages
            r'/return', r'/refund', r'/about', r'/contact',  # Other pages
            r'^https?://(?!www\.)[^/]+\.',  # Subdomains
            r'/page/\d+', r'/page-\d+', r'\?page=\d+',  # Paginated
            r'/\d{4}/\d{2}/\d{2}', r'/\d{4}/\d{2}',  # Archive pages
            r'/tag/', r'/category/', r'/author/', r'/search', r'/archive'  # Other
        ]
        
        for pattern in patterns:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    @staticmethod
    def filter_data(df):
        """Apply URL filtering to dataframe"""
        mask = ~df['page'].apply(URLFilter.should_filter_url)
        filtered_df = df[mask].copy()
        
        # Ensure URLs start with http
        filtered_df = filtered_df[filtered_df['page'].str.startswith('http')]
        
        return filtered_df
