"""
Enhanced URLCleaner with diagnostic capabilities to prevent zero-results issues.
"""
import re
import logging
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

class URLCleaner:
    def __init__(self, config):
        self.config = config
        self.excluded_parameters = set(config.get('excluded_parameters', []))
        self.excluded_pages = config.get('excluded_pages', [])
        self.excluded_patterns = config.get('excluded_patterns', [])
        
        # Compile patterns with error handling
        self.compiled_patterns = []
        for pattern in self.excluded_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error(f"INVALID REGEX PATTERN '{pattern}': {e}")
    
    def clean_dataframe(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """Enhanced cleaning with comprehensive diagnostics."""
        if df.empty:
            logger.warning("Empty dataframe provided to URLCleaner")
            return df
            
        original_count = len(df)
        logger.info(f"URLCleaner processing {original_count} rows")
        
        # Show sample URLs BEFORE filtering
        for col in url_columns:
            if col in df.columns:
                samples = df[col].head(3).tolist()
                logger.info(f"Sample URLs in '{col}': {samples}")
        
        # Apply filtering logic (simplified for safety)
        filtered_df = self._safe_filter(df, url_columns)
        
        final_count = len(filtered_df)
        removed = original_count - final_count
        percentage = (removed / original_count * 100) if original_count > 0 else 0
        
        # CRITICAL DIAGNOSTICS
        logger.info(f"URLCleaner Results:")
        logger.info(f"  Input: {original_count} rows")
        logger.info(f"  Output: {final_count} rows") 
        logger.info(f"  Filtered: {removed} rows ({percentage:.1f}%)")
        
        # ERROR CONDITIONS
        if percentage > 50:
            logger.error(f"⚠️ HIGH FILTER RATE: {percentage:.1f}% - Check config!")
            
        if final_count == 0:
            logger.error("❌ ALL URLS FILTERED! Analysis will fail!")
            logger.error("Your URL exclusion rules are too aggressive.")
            # Show what got filtered
            for col in url_columns:
                if col in df.columns:
                    logger.error(f"Filtered URLs from {col}: {df[col].head(5).tolist()}")
        
        return filtered_df
    
    def _safe_filter(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """Conservative filtering to prevent zero results."""
        
        def is_admin_url(url):
            """Check if URL is clearly an admin/system URL."""
            if pd.isna(url):
                return False
                
            url_str = str(url).lower()
            
            # Only exclude obvious admin paths
            admin_indicators = ['/wp-admin/', '/admin/', '/login', '/logout']
            return any(indicator in url_str for indicator in admin_indicators)
        
        # Create conservative exclusion mask
        exclusion_mask = pd.Series(False, index=df.index)
        
        for col in url_columns:
            if col in df.columns:
                col_exclusions = df[col].apply(is_admin_url)
                exclusion_mask = exclusion_mask | col_exclusions
        
        return df[~exclusion_mask].copy()
