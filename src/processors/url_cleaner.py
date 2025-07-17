import re
import logging
from urllib.parse import urlparse
from src.utils.config import Config

logger = logging.getLogger(__name__)

class URLCleaner:
    """Cleans and filters URLs based on configuration rules"""
    
    def __init__(self, config: Config):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.compiled_patterns = []
        for pattern in self.config.excluded_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    
    def should_exclude_url(self, url):
        """Check if URL should be excluded based on all rules"""
        if not url:
            return True
        
        url_lower = url.lower()
        
        # Check for excluded parameters
        for param in self.config.excluded_parameters:
            if param in url:
                return True
        
        # Check for excluded page patterns
        for page_pattern in self.config.excluded_pages:
            if page_pattern.lower() in url_lower:
                return True
        
        # Check regex patterns
        for pattern in self.compiled_patterns:
            if pattern.search(url):
                return True
        
        # Check for subdomains
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                # Count dots in hostname - more than expected means subdomain
                main_domain_dots = parsed.hostname.count('.') - 1  # Subtract 1 for .com, .org, etc.
                if main_domain_dots > 1:  # www.domain.com would be 1
                    return True
        except:
            pass
        
        return False
    
    def clean_dataframe(self, df, url_columns):
        """Remove rows with excluded URLs from dataframe"""
        initial_count = len(df)
        
        # Create mask for valid URLs
        mask = True
        for col in url_columns:
            if col in df.columns:
                mask = mask & ~df[col].apply(self.should_exclude_url)
        
        df_cleaned = df[mask].copy()
        
        removed_count = initial_count - len(df_cleaned)
        logger.info(f"Removed {removed_count} rows with excluded URLs ({removed_count/initial_count*100:.1f}%)")
        
        return df_cleaned
