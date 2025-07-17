"""
URL filtering processor for removing unwanted URLs from analysis
"""

import re
import pandas as pd


class URLFilter:
    """Handles URL filtering based on various rules"""
    
    def __init__(self):
        # URL patterns to exclude
        self.special_chars = ['=', '?', '#', '&']
        
        # Legal and navigation page patterns
        self.legal_patterns = [
            r'privacy[\s-]?policy', r'terms[\s-]?of[\s-]?service',
            r'terms[\s-]?and[\s-]?conditions', r'shipping[\s-]?policy',
            r'return[\s-]?policy', r'refund[\s-]?policy', r'cookie[\s-]?policy',
            r'disclaimer', r'legal', r'gdpr', r'ccpa'
        ]
        
        self.nav_patterns = [
            r'about[\s-]?us', r'contact[\s-]?us', r'contact', r'about',
            r'faq', r'help', r'support', r'careers', r'jobs'
        ]
        
        # Technical patterns
        self.technical_patterns = [
            r'/page/\d+', r'/page-\d+', r'/p\d+', r'/archive/',
            r'/\d{4}/\d{2}/\d{2}', r'/\d{4}/\d{2}', r'/tag/', r'/category/',
            r'/author/', r'/search/', r'/feed/', r'/wp-content/', r'/wp-admin/',
            r'/wp-includes/', r'/wp-json/', r'/xmlrpc.php', r'/trackback/',
            r'/comments/', r'/comment-page-\d+', r'/attachment/', r'/embed/'
        ]
    
    def should_exclude_url(self, url: str) -> bool:
        """Determine if a URL should be excluded from analysis"""
        url_lower = url.lower()
        
        # Check for special characters
        for char in self.special_chars:
            if char in url:
                return True
        
        # Check for subdomains (excluding www)
        if self._has_subdomain(url):
            return True
        
        # Check for legal pages
        for pattern in self.legal_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Check for navigation pages
        for pattern in self.nav_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Check for technical patterns
        for pattern in self.technical_patterns:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    def _has_subdomain(self, url: str) -> bool:
        """Check if URL has subdomain (excluding www)"""
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            parts = domain.split('.')
            if len(parts) > 2 and parts[0] != 'www':
                return True
        return False
    
    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to exclude unwanted URLs"""
        mask = ~df['page'].apply(self.should_exclude_url)
        return df[mask].copy()
