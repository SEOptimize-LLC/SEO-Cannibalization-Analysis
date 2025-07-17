"""
Column mapping utilities for flexible column name handling
"""

import pandas as pd
from typing import Dict, List, Tuple

# Comprehensive column name mappings for GSC data
GSC_COLUMN_MAPPINGS = {
    'query': [
        'query', 'queries', 'search_query', 'search queries', 'search_query', 'search_queries',
        'keyword', 'keywords', 'search_term', 'search_terms', 'search term', 'search terms',
        'QUERY', 'QUERIES', 'KEYWORD', 'KEYWORDS'
    ],
    'page': [
        'page', 'pages', 'url', 'urls', 'landing_page', 'landing pages', 'landing_page', 'landing_pages',
        'landing page', 'landing pages', 'PAGE', 'URL', 'LANDING_PAGE', 'LANDING PAGE'
    ],
    'clicks': [
        'clicks', 'click', 'total_clicks', 'total clicks', 'clicks_sum', 'CLICKS', 'CLICK', 'TOTAL_CLICKS'
    ],
    'impressions': [
        'impressions', 'impression', 'total_impressions', 'total impressions', 'impressions_sum',
        'IMPRESSIONS', 'IMPRESSION', 'TOTAL_IMPRESSIONS'
    ],
    'position': [
        'position', 'positions', 'avg_position', 'average_position', 'avg position', 'average position',
        'POSITION', 'AVG_POSITION', 'AVERAGE_POSITION'
    ]
}

# Semantic similarity column mappings
SIMILARITY_COLUMN_MAPPINGS = {
    'primary_url': [
        'address', 'url', 'page', 'primary_url', 'url1', 'page1',
        'ADDRESS', 'URL', 'PAGE', 'PRIMARY_URL'
    ],
    'secondary_url': [
        'closest_semantically_similar_address', 'closest_url', 'similar_url', 'secondary_url', 'url2', 'page2',
        'CLOSEST_SEMANTICALLY_SIMILAR_ADDRESS', 'CLOSEST_URL', 'SECONDARY_URL'
    ],
    'similarity_score': [
        'semantic_similarity_score', 'similarity', 'similarity_score', 'score',
        'SEMANTIC_SIMILARITY_SCORE', 'SIMILARITY', 'SIMILARITY_SCORE', 'SCORE'
    ]
}

def normalize_column_names(df: pd.DataFrame, column_mappings: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Normalize column names based on provided mappings
    
    Args:
        df: Input DataFrame
        column_mappings: Dictionary mapping standard names to possible variations
    
    Returns:
        DataFrame with normalized column names
    """
    df_copy = df.copy()
    column_mapping = {}
    
    # Create reverse mapping for easy lookup
    reverse_mapping = {}
    for standard_name, variations in column_mappings.items():
        for variation in variations:
            reverse_mapping[variation.lower()] = standard_name
    
    # Map actual columns to standard names
    for col in df_copy.columns:
        col_lower = col.lower().strip()
        if col_lower in reverse_mapping:
            column_mapping[col] = reverse_mapping[col_lower]
    
    return df_copy.rename(columns=column_mapping)

def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that required columns are present
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing
