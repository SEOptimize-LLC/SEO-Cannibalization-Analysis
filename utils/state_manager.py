"""
Session State Management for Streamlit App
Handles persistent state across page refreshes and interactions
"""

import streamlit as st
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime

class StateManager:
    """Centralized state management for the application"""
    
    # Define default state values
    DEFAULT_STATE = {
        # Data state
        'data_loaded': False,
        'df': None,
        'processed_df': None,
        
        # Authentication state
        'gsc_authenticated': False,
        'gsc_credentials': None,
        'selected_property': None,
        
        # Analysis state
        'analysis_complete': False,
        'analysis_results': {},
        'analysis_timestamp': None,
        
        # Configuration state
        'config': {
            'click_threshold': 0.05,
            'min_clicks': 10,
            'brand_terms': [],
            'enable_intent_detection': True,
            'enable_serp_analysis': True,
            'enable_content_gap': True,
            'enable_ml_insights': True
        },
        
        # UI state
        'selected_query': None,
        'selected_tab': 0,
        'export_format': None,
        
        # Cache state
        'cache': {},
        'cache_timestamps': {}
    }
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state with default values"""
        for key, value in StateManager.DEFAULT_STATE.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any):
        """Set value in session state"""
        st.session_state[key] = value
    
    @staticmethod
    def update_config(config_updates: Dict):
        """Update configuration values"""
        if 'config' not in st.session_state:
            st.session_state.config = StateManager.DEFAULT_STATE['config'].copy()
        
        st.session_state.config.update(config_updates)
    
    @staticmethod
    def get_config(key: str = None) -> Any:
        """Get configuration value or entire config"""
        if key:
            return st.session_state.config.get(key)
        return st.session_state.config
    
    @staticmethod
    def set_data(df: pd.DataFrame):
        """Set the main dataframe and update related states"""
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = {}
    
    @staticmethod
    def get_data() -> Optional[pd.DataFrame]:
        """Get the main dataframe"""
        return st.session_state.get('df')
    
    @staticmethod
    def set_analysis_results(results: Dict):
        """Set analysis results and update timestamp"""
        st.session_state.analysis_results = results
        st.session_state.analysis_complete = True
        st.session_state.analysis_timestamp = datetime.now()
    
    @staticmethod
    def get_analysis_results() -> Dict:
        """Get analysis results"""
        return st.session_state.get('analysis_results', {})
    
    @staticmethod
    def cache_result(key: str, value: Any, ttl: int = 3600):
        """Cache a result with TTL"""
        st.session_state.cache[key] = value
        st.session_state.cache_timestamps[key] = datetime.now()
    
    @staticmethod
    def get_cached_result(key: str, ttl: int = 3600) -> Optional[Any]:
        """Get cached result if not expired"""
        if key not in st.session_state.cache:
            return None
        
        # Check if cache is expired
        timestamp = st.session_state.cache_timestamps.get(key)
        if timestamp and (datetime.now() - timestamp).seconds > ttl:
            # Cache expired, remove it
            del st.session_state.cache[key]
            del st.session_state.cache_timestamps[key]
            return None
        
        return st.session_state.cache[key]
    
    @staticmethod
    def clear_cache():
        """Clear all cached results"""
        st.session_state.cache = {}
        st.session_state.cache_timestamps = {}
    
    @staticmethod
    def reset_analysis():
        """Reset analysis state"""
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = {}
        st.session_state.analysis_timestamp = None
        StateManager.clear_cache()
    
    @staticmethod
    def is_data_loaded() -> bool:
        """Check if data is loaded"""
        return st.session_state.get('data_loaded', False)
    
    @staticmethod
    def is_analysis_complete() -> bool:
        """Check if analysis is complete"""
        return st.session_state.get('analysis_complete', False)
    
    @staticmethod
    def export_state() -> Dict:
        """Export current state for debugging or backup"""
        return {
            'data_loaded': st.session_state.get('data_loaded'),
            'analysis_complete': st.session_state.get('analysis_complete'),
            'config': st.session_state.get('config'),
            'analysis_timestamp': st.session_state.get('analysis_timestamp'),
            'cache_size': len(st.session_state.get('cache', {}))
        }


# Convenience function for initialization
def initialize_session_state():
    """Initialize session state - convenience function"""
    StateManager.initialize_session_state()


# Decorators for state management
def require_data(func):
    """Decorator to ensure data is loaded before function execution"""
    def wrapper(*args, **kwargs):
        if not StateManager.is_data_loaded():
            st.warning("Please load data first in the Data Input tab.")
            return None
        return func(*args, **kwargs)
    return wrapper


def require_analysis(func):
    """Decorator to ensure analysis is complete before function execution"""
    def wrapper(*args, **kwargs):
        if not StateManager.is_analysis_complete():
            st.warning("Please run the analysis first in the Analysis tab.")
            return None
        return func(*args, **kwargs)
    return wrapper


def cache_analysis(ttl: int = 3600):
    """Decorator to cache analysis results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Check cache
            cached_result = StateManager.get_cached_result(cache_key, ttl)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            StateManager.cache_result(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
