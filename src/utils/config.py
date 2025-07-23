"""
Enhanced Config class with dictionary-style interface support
for enterprise SEO cannibalization analysis.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """
    Enterprise configuration management with dual access patterns:
    - Property-based access (config.url_filters.excluded_parameters)
    - Dictionary-style access (config.get('excluded_parameters', []))
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config_data = {}
        self._load_config()
        self._setup_properties()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with error handling."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration structure."""
        self.config_data = {
            'url_filters': {
                'excluded_parameters': [
                    'utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid'
                ],
                'excluded_pages': [
                    '/wp-admin/', '/admin/', '/login', '/logout'
                ],
                'excluded_patterns': [
                    '^.*/wp-admin/.*$',
                    '^.*/feed/?$'
                ]
            },
            'analysis': {
                'similarity_thresholds': {'high': 0.90, 'medium': 0.85},
                'priority_percentiles': {'high': 75, 'medium': 40},
                'min_clicks': 1,
                'min_impressions': 10,
                'min_queries': 1
            }
        }
    
    def _setup_properties(self) -> None:
        """Setup property-based access for configuration sections."""
        # URL filters configuration
        url_filters = self.config_data.get('url_filters', {})
        self.url_filters = ConfigSection(url_filters)
        
        # Analysis configuration  
        analysis = self.config_data.get('analysis', {})
        self.analysis = ConfigSection(analysis)
    
    # Dictionary-style interface methods
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dictionary-style access to configuration values.
        Supports nested key access with dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'url_filters.excluded_parameters')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Handle nested keys with dot notation
            if '.' in key:
                keys = key.split('.')
                value = self.config_data
                for k in keys:
                    value = value.get(k, {})
                return value if value != {} else default
            else:
                # Direct key access for backward compatibility
                return self.config_data.get(key, default)
        except Exception as e:
            logger.warning(f"Error accessing config key '{key}': {e}")
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Support bracket notation access: config['key']"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return value
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: 'key' in config"""
        return self.get(key) is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config_data.copy()
    
    def save(self) -> None:
        """Save current configuration to YAML file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")


class ConfigSection:
    """
    Configuration section wrapper for property-based access.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        # Set attributes dynamically
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for section data."""
        return self._data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Support bracket notation."""
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return key in self._data
