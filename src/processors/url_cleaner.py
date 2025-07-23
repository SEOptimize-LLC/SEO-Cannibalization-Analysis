"""
Enhanced URLCleaner with legal page filtering and parameter cleanup.
"""
import re
import logging
from typing import List, Dict, Set
import pandas as pd
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class URLCleaner:
    def __init__(self, config):
        self.config = config
        self.excluded_parameters = set(config.get('excluded_parameters', []))
        self.excluded_pages = [page.lower() for page in config.get('excluded_pages', [])]
        self.excluded_patterns = config.get('excluded_patterns', [])
        
        # Compile patterns with error handling
        self.compiled_patterns = []
        for pattern in self.excluded_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error(f"INVALID REGEX PATTERN '{pattern}': {e}")
    
    def clean_dataframe(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """Enhanced cleaning with parameter and legal page filtering."""
        if df.empty:
            logger.warning("Empty dataframe provided to URLCleaner")
            return df
            
        original_count = len(df)
        logger.info(f"URLCleaner processing {original_count} rows")
        
        # Apply enhanced filtering
        filtered_df = self._enhanced_filter(df, url_columns)
        
        final_count = len(filtered_df)
        removed = original_count - final_count
        percentage = (removed / original_count * 100) if original_count > 0 else 0
        
        # Enhanced diagnostics
        logger.info(f"URLCleaner Results:")
        logger.info(f"  Input: {original_count} rows")
        logger.info(f"  Output: {final_count} rows") 
        logger.info(f"  Filtered: {removed} rows ({percentage:.1f}%)")
        
        # Safety checks
        if percentage > 80:
            logger.error(f"⚠️ VERY HIGH FILTER RATE: {percentage:.1f}%")
        elif percentage > 60:
            logger.warning(f"⚠️ HIGH FILTER RATE: {percentage:.1f}%")
        
        if final_count == 0:
            logger.error("❌ ALL URLS FILTERED! Check configuration!")
        
        return filtered_df
    
    def _enhanced_filter(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """Enhanced filtering with parameter and legal page detection."""
        
        def should_exclude_url(url):
            """Comprehensive URL exclusion logic."""
            if pd.isna(url):
                return False
                
            url_str = str(url).lower().strip()
            
            # 1. Check for excluded parameter characters
            if self._has_excluded_parameters(url_str):
                return True
            
            # 2. Check for legal/internal pages
            if self._is_legal_page(url_str):
                return True
            
            # 3. Check for admin URLs
            if self._is_admin_url(url_str):
                return True
            
            # 4. Check regex patterns
            if self._matches_excluded_pattern(url_str):
                return True
            
            return False
        
        # Apply filtering to all URL columns
        exclusion_mask = pd.Series(False, index=df.index)
        
        for col in url_columns:
            if col in df.columns:
                col_exclusions = df[col].apply(should_exclude_url)
                exclusion_mask = exclusion_mask | col_exclusions
        
        return df[~exclusion_mask].copy()
    
    def _has_excluded_parameters(self, url: str) -> bool:
        """Check if URL contains excluded parameter characters."""
        # Check for basic parameter characters
        parameter_chars = {'?', '=', '#', '&'}
        excluded_chars = parameter_chars.intersection(self.excluded_parameters)
        
        if excluded_chars:
            return any(char in url for char in excluded_chars)
        
        # Check for specific parameter names
        try:
            parsed = urlparse(url)
            if parsed.query:
                params = parse_qs(parsed.query)
                param_names = set(params.keys())
                excluded_params = param_names.intersection(self.excluded_parameters)
                return len(excluded_params) > 0
        except:
            pass
        
        return False
    
    def _is_legal_page(self, url: str) -> bool:
        """Check if URL is a legal/internal page with no SEO value."""
        # Direct string matching for legal pages
        for excluded_page in self.excluded_pages:
            if excluded_page in url:
                return True
        
        # Path-based checking
        try:
            parsed = urlparse(url)
            path = parsed.path.lower().strip('/')
            
            # Check if path matches legal page patterns
            legal_patterns = [
                'privacy', 'terms', 'legal', 'disclaimer', 'cookie',
                'about', 'contact', 'shipping', 'return'
            ]
            
            return any(pattern in path for pattern in legal_patterns)
        except:
            pass
        
        return False
    
    def _is_admin_url(self, url: str) -> bool:
        """Check if URL is an admin/system URL."""
        admin_indicators = ['/wp-admin/', '/admin/', '/login', '/logout']
        return any(indicator in url for indicator in admin_indicators)
    
    def _matches_excluded_pattern(self, url: str) -> bool:
        """Check if URL matches any excluded regex pattern."""
        for pattern in self.compiled_patterns:
            try:
                if pattern.search(url):
                    return True
            except:
                continue
        return False
