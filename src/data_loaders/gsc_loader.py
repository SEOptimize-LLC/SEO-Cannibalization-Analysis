"""
Google Search Console data loader with flexible column handling
"""

import pandas as pd
from typing import Optional
from ..utils.column_mapper import normalize_column_names, validate_required_columns, GSC_COLUMN_MAPPINGS


class GSCLoader:
    """Handles loading and initial processing of Google Search Console data"""
    
    def __init__(self):
        self.required_columns = ['query', 'page', 'clicks', 'impressions']
    
    def load(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load GSC data from file
        
        Args:
            file_path: Path to the GSC file
            file_type: 'csv' or 'xlsx'
        
        Returns:
            DataFrame with normalized column names
        """
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_type.lower() == 'xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Normalize column names
            df = normalize_column_names(df, GSC_COLUMN_MAPPINGS)
            
            # Validate required columns
            is_valid, missing = validate_required_columns(df, self.required_columns)
            if not is_valid:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Ensure numeric columns are properly typed
            numeric_cols = ['clicks', 'impressions', 'position']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN in required columns
            df = df.dropna(subset=self.required_columns)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading GSC data: {str(e)}")
    
    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate GSC data to get metrics per URL
        
        Args:
            df: GSC DataFrame
            
        Returns:
            DataFrame with URL-level metrics
        """
        return df.groupby('page').agg({
            'query': 'nunique',  # Count unique queries
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index().rename(columns={
            'query': 'indexed_queries',
            'clicks': 'total_clicks',
            'impressions': 'total_impressions'
        })
