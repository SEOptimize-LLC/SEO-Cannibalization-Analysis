"""
Application Configuration and Settings
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application settings
APP_CONFIG = {
    'app_name': 'SEO Cannibalization Analyzer',
    'app_icon': '🔍',
    'app_description': 'Advanced Detection & Resolution Tool',
    'version': '2.0.0',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Analysis thresholds
ANALYSIS_DEFAULTS = {
    'click_threshold': 0.05,  # 5% default
    'min_clicks': 10,
    'min_pages': 2,
    'intent_mismatch_threshold': 0.5,
    'content_similarity_threshold': 0.7,
    'traffic_recovery_estimate': 0.7  # 70% recovery estimate
}

# Google Search Console settings
GSC_CONFIG = {
    'client_id': os.getenv('GSC_CLIENT_ID', ''),
    'client_secret': os.getenv('GSC_CLIENT_SECRET', ''),
    'redirect_uri': os.getenv('GSC_REDIRECT_URI', 'http://localhost:8501'),
    'scopes': ['https://www.googleapis.com/auth/webmasters.readonly'],
    'default_date_range_days': 90
}

# Export settings
EXPORT_CONFIG = {
    'formats': ['Excel', 'CSV', 'JSON'],
    'integrations': ['Screaming Frog', 'Ahrefs', 'SEMrush', 'Sitebulb'],
    'max_export_rows': 50000
}

# Visualization settings
VIZ_CONFIG = {
    'color_scheme': {
        'primary': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    },
    'chart_height': 400,
    'chart_width': None  # Use container width
}

# Feature flags
FEATURES = {
    'intent_detection': True,
    'serp_analysis': True,
    'content_gap_analysis': True,
    'ml_insights': True,
    'competitive_analysis': True,
    'internal_link_analysis': True
}

# API endpoints (for future external APIs)
API_ENDPOINTS = {
    'serp_api': os.getenv('SERP_API_ENDPOINT', ''),
    'content_api': os.getenv('CONTENT_API_ENDPOINT', ''),
    'ml_api': os.getenv('ML_API_ENDPOINT', '')
}

# Cache settings
CACHE_CONFIG = {
    'ttl': 3600,  # 1 hour
    'max_entries': 1000
}
