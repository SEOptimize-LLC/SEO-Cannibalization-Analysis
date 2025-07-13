import pandas as pd
import re

def normalize_column_names(df):
    """
    Normalize column names to match expected format: page, query, clicks, impressions, position
    Handles various naming conventions including case variations, plural/singular, and different wording
    
    Args:
        df (pd.DataFrame): Input DataFrame with potentially varied column names
        
    Returns:
        tuple: (normalized_df, mapping_dict) - DataFrame with normalized columns and applied mappings
    """
    
    # Define mapping patterns for each expected column
    column_mappings = {
        'page': [
            'page', 'pages', 'url', 'urls', 'landing page', 'landing pages', 
            'landingpage', 'landingpages', 'page_url', 'page url', 'pageurl',
            'destination', 'destinations', 'link', 'links', 'webpage', 'webpages',
            'site', 'sites', 'page_path', 'pagepath', 'path'
        ],
        'query': [
            'query', 'queries', 'keyword', 'keywords', 'search term', 'search terms',
            'searchterm', 'searchterms', 'search query', 'search queries',
            'searchquery', 'searchqueries', 'term', 'terms', 'phrase', 'phrases',
            'search_query', 'search_term', 'top_queries', 'top queries'
        ],
        'clicks': [
            'clicks', 'click', 'total clicks', 'totalclicks', 'click count',
            'clickcount', 'click_count', 'ctr clicks', 'ctrclicks', 'total_clicks'
        ],
        'impressions': [
            'impressions', 'impression', 'total impressions', 'totalimpressions',
            'impression count', 'impressioncount', 'impression_count',
            'views', 'view', 'total views', 'totalviews', 'total_impressions'
        ],
        'position': [
            'position', 'positions', 'avg position', 'avgposition', 'avg_position',
            'average position', 'averageposition', 'average_position',
            'avg. position', 'avg. pos', 'avg pos', 'avgpos', 'avg.pos',
            'rank', 'ranking', 'rankings', 'avg rank', 'avgrank', 'avg_rank',
            'average rank', 'averagerank', 'average_rank', 'avg_ranking'
        ]
    }
    
    # Create a normalized copy of the dataframe
    df_normalized = df.copy()
    
    # Get current column names (normalized to lowercase for comparison)
    current_columns = [col.lower().strip() for col in df.columns]
    
    # Track which columns were mapped
    mapped_columns = {}
    
    # For each expected column, find the best match
    for expected_col, variations in column_mappings.items():
        # Normalize variations to lowercase
        normalized_variations = [var.lower().strip() for var in variations]
        
        # Find exact match first
        for i, current_col in enumerate(current_columns):
            if current_col in normalized_variations:
                original_col_name = df.columns[i]
                mapped_columns[original_col_name] = expected_col
                break
        
        # If no exact match found, try partial matches
        if expected_col not in mapped_columns.values():
            for i, current_col in enumerate(current_columns):
                for variation in normalized_variations:
                    # Check if variation is contained in current column or vice versa
                    if (variation in current_col or current_col in variation) and len(variation) > 2:
                        original_col_name = df.columns[i]
                        mapped_columns[original_col_name] = expected_col
                        break
                if expected_col in mapped_columns.values():
                    break
    
    # Rename columns using the mapping
    df_normalized.rename(columns=mapped_columns, inplace=True)
    
    return df_normalized, mapped_columns

def validate_required_columns(df, show_mappings=False):
    """
    Validate that all required columns are present after normalization
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        show_mappings (bool): Whether to return mapping information
        
    Returns:
        tuple: (is_valid, missing_columns, mapped_columns_info)
    """
    required_cols = ['page', 'query', 'clicks', 'impressions', 'position']
    
    # Normalize column names
    df_normalized, mappings = normalize_column_names(df)
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df_normalized.columns]
    
    is_valid = len(missing_cols) == 0
    
    if show_mappings:
        return is_valid, missing_cols, mappings, df_normalized
    else:
        return is_valid, missing_cols, df_normalized
