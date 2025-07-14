"""
Column mapping utilities for SEO Cannibalization Analysis
Handles column normalization and validation for Google Search Console data
"""

import pandas as pd
from typing import List, Dict, Tuple

# Standard column mappings for Google Search Console data
COLUMN_MAPPINGS = {
    # Page/URL variations
    'page': 'page',
    'url': 'page',
    'landing_page': 'page',
    'landing page': 'page',
    'pages': 'page',
    
    # Query/Keyword variations
    'query': 'query',
    'keyword': 'query',
    'search_query': 'query',
    'search query': 'query',
    'search_term': 'query',
    'search term': 'query',
    'queries': 'query',
    'keywords': 'query',
    
    # Clicks variations
    'clicks': 'clicks',
    'click': 'clicks',
    'total_clicks': 'clicks',
    'total clicks': 'clicks',
    
    # Impressions variations
    'impressions': 'impressions',
    'impression': 'impressions',
    'total_impressions': 'impressions',
    'total impressions': 'impressions',
    
    # Position variations
    'position': 'position',
    'average_position': 'position',
    'average position': 'position',
    'avg_position': 'position',
    'avg position': 'position',
    'ranking': 'position',
    'rank': 'position'
}

# Required columns for the analysis
REQUIRED_COLUMNS = ['page', 'query', 'clicks', 'impressions', 'position']


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard format.
    
    Args:
        df: DataFrame with potentially varied column names
        
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying the original
    df_normalized = df.copy()
    
    # Convert all column names to lowercase for matching
    current_columns = {col: col.lower().strip() for col in df_normalized.columns}
    
    # Map columns to standard names
    rename_dict = {}
    for original_col, lower_col in current_columns.items():
        if lower_col in COLUMN_MAPPINGS:
            standard_name = COLUMN_MAPPINGS[lower_col]
            if standard_name != original_col:
                rename_dict[original_col] = standard_name
    
    # Rename columns
    if rename_dict:
        df_normalized = df_normalized.rename(columns=rename_dict)
    
    return df_normalized


def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns are present in the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_columns = []
    
    for required_col in REQUIRED_COLUMNS:
        if required_col not in df.columns:
            missing_columns.append(required_col)
    
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns


def get_column_mapping_report(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate a report about column mapping and validation.
    
    Args:
        df: Original DataFrame
        
    Returns:
        Dictionary containing mapping details
    """
    original_columns = list(df.columns)
    normalized_df = normalize_column_names(df)
    normalized_columns = list(normalized_df.columns)
    
    is_valid, missing_columns = validate_required_columns(normalized_df)
    
    # Identify which columns were mapped
    mapped_columns = {}
    for orig, norm in zip(original_columns, normalized_columns):
        if orig != norm:
            mapped_columns[orig] = norm
    
    # Identify unmapped columns
    unmapped_columns = [col for col in normalized_columns 
                       if col not in REQUIRED_COLUMNS]
    
    report = {
        'original_columns': original_columns,
        'normalized_columns': normalized_columns,
        'mapped_columns': mapped_columns,
        'unmapped_columns': unmapped_columns,
        'missing_required_columns': missing_columns,
        'is_valid': is_valid,
        'validation_message': 'All required columns present' if is_valid 
                            else f'Missing required columns: {", ".join(missing_columns)}'
    }
    
    return report


def prepare_gsc_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare Google Search Console data for analysis.
    
    Args:
        df: Raw DataFrame from GSC export
        verbose: Whether to print validation messages
        
    Returns:
        Normalized DataFrame ready for analysis
        
    Raises:
        ValueError: If required columns are missing
    """
    # Normalize column names
    df_normalized = normalize_column_names(df)
    
    # Validate required columns
    is_valid, missing_columns = validate_required_columns(df_normalized)
    
    if not is_valid:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}\n"
        error_msg += f"Available columns: {', '.join(df_normalized.columns)}\n"
        error_msg += "Please ensure your CSV has columns for: page/url, query/keyword, clicks, impressions, and position"
        raise ValueError(error_msg)
    
    if verbose:
        print("✓ Column validation successful")
        print(f"  - Found {len(df_normalized)} rows")
        print(f"  - Columns: {', '.join(df_normalized.columns)}")
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['clicks', 'impressions', 'position']
    for col in numeric_columns:
        df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
    
    # Remove rows with invalid numeric values
    initial_rows = len(df_normalized)
    df_normalized = df_normalized.dropna(subset=numeric_columns)
    
    if verbose and initial_rows != len(df_normalized):
        print(f"  - Removed {initial_rows - len(df_normalized)} rows with invalid numeric values")
    
    return df_normalized
