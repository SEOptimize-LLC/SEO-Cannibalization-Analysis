"""
Enhanced URLCleaner with proper Config object integration.
Fixed AttributeError for enterprise Config class interface.
"""
import re
import logging
from typing import List, Set
import pandas as pd

logger = logging.getLogger(__name__)

class URLCleaner:
    """
    Enterprise URL cleaning and filtering processor.
    Compatible with both dictionary-style and property-based Config objects.
    """
    
    def __init__(self, config):
        """
        Initialize URLCleaner with configuration.
        
        Args:
            config: Config object (supports both dict-style and property access)
        """
        self.config = config
        
        # Extract URL filter configuration with fallback patterns
        self.excluded_parameters = self._extract_config_list('url_filters.excluded_parameters')
        self.excluded_pages = self._extract_config_list('url_filters.excluded_pages')
        self.excluded_patterns = self._extract_config_list('url_filters.excluded_patterns')
        
        # Compile regex patterns with error handling
        self.compiled_patterns = []
        for pattern in self.excluded_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
        
        logger.info(f"URLCleaner initialized with {len(self.excluded_parameters)} parameter filters, "
                   f"{len(self.excluded_pages)} page filters, and {len(self.compiled_patterns)} pattern filters")
    
    def _extract_config_list(self, key: str) -> List[str]:
        """
        Extract configuration list with multiple access pattern support.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            List of configuration values
        """
        try:
            # Try dictionary-style access first
            if hasattr(self.config, 'get'):
                value = self.config.get(key, [])
                if value:
                    return value
            
            # Try property-based access
            if '.' in key:
                keys = key.split('.')
                obj = self.config
                for k in keys:
                    if hasattr(obj, k):
                        obj = getattr(obj, k)
                    else:
                        logger.warning(f"Config property '{k}' not found in path '{key}'")
                        return []
                
                # Handle both list and ConfigSection objects
                if isinstance(obj, list):
                    return obj
                elif hasattr(obj, '_data') and isinstance(obj._data, list):
                    return obj._data
                else:
                    logger.warning(f"Config value at '{key}' is not a list: {type(obj)}")
                    return []
            
            # Direct property access fallback
            if hasattr(self.config, key):
                value = getattr(self.config, key)
                return value if isinstance(value, list) else []
            
            logger.warning(f"Configuration key '{key}' not found, using empty list")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting config list for '{key}': {e}")
            return []
    
    def clean_dataframe(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """
        Clean DataFrame by filtering URLs based on configuration rules.
        
        Args:
            df: Input DataFrame
            url_columns: List of column names containing URLs
            
        Returns:
            Cleaned DataFrame with filtered URLs
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to URLCleaner")
            return df
        
        original_count = len(df)
        logger.info(f"URLCleaner processing {original_count} rows")
        
        # Apply filtering logic
        filtered_df = self._apply_url_filters(df, url_columns)
        
        final_count = len(filtered_df)
        removed = original_count - final_count
        percentage = (removed / original_count * 100) if original_count > 0 else 0
        
        # Log results with warnings for high filter rates
        logger.info(f"URLCleaner Results: {original_count} → {final_count} rows "
                   f"({removed} filtered, {percentage:.1f}%)")
        
        if percentage > 80:
            logger.error(f"⚠️ CRITICAL: {percentage:.1f}% of URLs filtered - check configuration!")
        elif percentage > 50:
            logger.warning(f"⚠️ HIGH FILTER RATE: {percentage:.1f}% - review exclusion rules")
        
        if final_count == 0:
            logger.error("❌ ALL URLS FILTERED! Analysis will fail. Check exclusion configuration.")
        
        return filtered_df
    
    def _apply_url_filters(self, df: pd.DataFrame, url_columns: List[str]) -> pd.DataFrame:
        """Apply URL filtering rules to DataFrame."""
        
        def should_exclude_url(url: str) -> bool:
            """Determine if URL should be excluded based on configuration rules."""
            if pd.isna(url) or not url.strip():
                return False
            
            url_str = str(url).lower().strip()
            
            # Check excluded pages (substring matching)
            for excluded_page in self.excluded_pages:
                if excluded_page.lower() in url_str:
                    return True
            
            # Check regex patterns
            for pattern in self.compiled_patterns:
                if pattern.search(url):
                    return True
            
            # Check parameters (basic implementation)
            for param in self.excluded_parameters:
                if param.lower() in url_str:
                    return True
            
            return False
        
        # Create exclusion mask for all URL columns
        exclusion_mask = pd.Series(False, index=df.index)
        
        for col in url_columns:
            if col in df.columns:
                col_exclusions = df[col].apply(should_exclude_url)
                exclusion_mask = exclusion_mask | col_exclusions
        
        return df[~exclusion_mask].copy()
