import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager for the SEO Cannibalization Tool"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def url_filters(self):
        return self.config.get('url_filters', {})
    
    @property
    def analysis_config(self):
        return self.config.get('analysis', {})
    
    @property
    def excluded_parameters(self):
        return self.url_filters.get('excluded_parameters', [])
    
    @property
    def excluded_pages(self):
        return self.url_filters.get('excluded_pages', [])
    
    @property
    def excluded_patterns(self):
        return self.url_filters.get('excluded_patterns', [])
    
    @property
    def similarity_thresholds(self):
        return self.analysis_config.get('similarity_thresholds', {})
