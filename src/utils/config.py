"""
Enhanced enterprise configuration management for SEO Cannibalization Analysis.
Supports both property-based and dictionary-style access patterns.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimilarityThresholds:
    """Similarity threshold configuration with dictionary-like access."""
    
    high: float = 0.90
    medium: float = 0.85
    low: float = 0.70
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for compatibility."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Support bracket notation access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in similarity thresholds")


@dataclass  
class PriorityPercentiles:
    """Priority percentile configuration."""
    
    high: int = 75
    medium: int = 40
    low: int = 10
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for compatibility."""
        return getattr(self, key, default)


@dataclass
class AnalysisConfig:
    """Analysis configuration section."""
    
    similarity_thresholds: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    priority_percentiles: PriorityPercentiles = field(default_factory=PriorityPercentiles)
    min_clicks: int = 1
    min_impressions: int = 10
    min_queries: int = 1
    min_similarity_score: float = 0.80


@dataclass
class URLFilters:
    """URL filtering configuration."""
    
    excluded_parameters: List[str] = field(default_factory=lambda: [
        'utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid'
    ])
    excluded_pages: List[str] = field(default_factory=lambda: [
        '/wp-admin/', '/admin/', '/login', '/logout'
    ])
    excluded_patterns: List[str] = field(default_factory=lambda: [
        '^.*/wp-admin/.*$', '^.*/feed/?$'
    ])


class Config:
    """
    Enterprise configuration management with dual access patterns.
    Supports both property-based access (config.analysis.similarity_thresholds) 
    and direct attribute access (config.similarity_thresholds).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file with fallback defaults."""
        self.config_path = Path(config_path)
        self._load_configuration()
        self._setup_enterprise_defaults()
    
    def _load_configuration(self) -> None:
        """Load configuration from YAML file with comprehensive error handling."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                config_data = {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            config_data = {}
        
        self._raw_config = config_data
    
    def _setup_enterprise_defaults(self) -> None:
        """Setup enterprise-grade configuration with proper defaults."""
        # Load analysis configuration
        analysis_data = self._raw_config.get('analysis', {})
        
        # Setup similarity thresholds
        similarity_data = analysis_data.get('similarity_thresholds', {})
        self.similarity_thresholds = SimilarityThresholds(
            high=similarity_data.get('high', 0.90),
            medium=similarity_data.get('medium', 0.85),
            low=similarity_data.get('low', 0.70)
        )
        
        # Setup priority percentiles  
        priority_data = analysis_data.get('priority_percentiles', {})
        self.priority_percentiles = PriorityPercentiles(
            high=priority_data.get('high', 75),
            medium=priority_data.get('medium', 40),
            low=priority_data.get('low', 10)
        )
        
        # Setup analysis configuration object
        self.analysis = AnalysisConfig(
            similarity_thresholds=self.similarity_thresholds,
            priority_percentiles=self.priority_percentiles,
            min_clicks=analysis_data.get('min_clicks', 1),
            min_impressions=analysis_data.get('min_impressions', 10),
            min_queries=analysis_data.get('min_queries', 1),
            min_similarity_score=analysis_data.get('min_similarity_score', 0.80)
        )
        
        # Setup URL filters
        url_filter_data = self._raw_config.get('url_filters', {})
        self.url_filters = URLFilters(
            excluded_parameters=url_filter_data.get('excluded_parameters', 
                ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']),
            excluded_pages=url_filter_data.get('excluded_pages',
                ['/wp-admin/', '/admin/', '/login', '/logout']),
            excluded_patterns=url_filter_data.get('excluded_patterns',
                ['^.*/wp-admin/.*$', '^.*/feed/?$'])
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access to configuration values."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._raw_config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Support bracket notation access."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self._raw_config:
            return self._raw_config[key]
        raise KeyError(f"Configuration key '{key}' not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary for serialization."""
        return {
            'analysis': {
                'similarity_thresholds': {
                    'high': self.similarity_thresholds.high,
                    'medium': self.similarity_thresholds.medium,
                    'low': self.similarity_thresholds.low
                },
                'priority_percentiles': {
                    'high': self.priority_percentiles.high,
                    'medium': self.priority_percentiles.medium,
                    'low': self.priority_percentiles.low
                },
                'min_clicks': self.analysis.min_clicks,
                'min_impressions': self.analysis.min_impressions,
                'min_queries': self.analysis.min_queries,
                'min_similarity_score': self.analysis.min_similarity_score
            },
            'url_filters': {
                'excluded_parameters': self.url_filters.excluded_parameters,
                'excluded_pages': self.url_filters.excluded_pages,
                'excluded_patterns': self.url_filters.excluded_patterns
            }
        }


# Global configuration instance for application use
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: str = "config.yaml") -> Config:
    """Force reload configuration from file."""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
