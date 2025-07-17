"""
Semantic similarity data loader with flexible column handling
"""

import pandas as pd
from ..utils.column_mapper import normalize_column_names, SIMILARITY_COLUMN_MAPPINGS


class SimilarityLoader:
    """Handles loading and processing of semantic similarity data"""
    
    def __init__(self):
        self.required_columns = ['primary_url', 'secondary_url', 'similarity_score']
    
    def load(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load similarity data from file
        
        Args:
            file_path: Path to the similarity file
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
            df = normalize_column_names(df, SIMILARITY_COLUMN_MAPPINGS)
            
            # Format similarity score to 1 decimal place
            if 'similarity_score' in df.columns:
                df['similarity_score'] = pd.to_numeric(
                    df['similarity_score'], 
                    errors='coerce'
                )
                # Handle percentage values (e.g., 89.45 -> 89.5)
                if df['similarity_score'].max() > 1:
                    df['similarity_score'] = df['similarity_score'] / 100
                df['similarity_score'] = df['similarity_score'].round(3)
            
            # Ensure required columns exist
            missing = [col for col in self.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Drop rows with NaN values
            df = df.dropna(subset=self.required_columns)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading similarity data: {str(e)}")
    
    def validate_urls_against_gsc(self, similarity_df: pd.DataFrame, 
                                gsc_urls: set) -> pd.DataFrame:
        """
        Filter out URLs that don't exist in GSC data
        
        Args:
            similarity_df: Similarity DataFrame
            gsc_urls: Set of URLs present in GSC data
            
        Returns:
            Filtered DataFrame with only valid URLs
        """
        mask = (
            similarity_df['primary_url'].isin(gsc_urls) & 
            similarity_df['secondary_url'].isin(gsc_urls)
        )
        return similarity_df[mask].copy()
