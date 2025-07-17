"""
Data loading and validation module
"""

import pandas as pd
from typing import Tuple


class DataLoader:
    """Handles loading and basic validation of input files"""
    
    @staticmethod
    def load_gsc(file) -> pd.DataFrame:
        """Load Google Search Console data"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Normalize column names
            column_mapping = {
                'Query': 'query', 'Search Query': 'query', 'search_query': 'query',
                'Page': 'page', 'URL': 'page', 'url': 'page', 'landing_page': 'page',
                'Clicks': 'clicks', 'clicks': 'clicks',
                'Impressions': 'impressions', 'impressions': 'impressions',
                'Position': 'position', 'Avg. Position': 'position', 'position': 'position'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() 
                                  if k in df.columns})
            
            required_cols = ['query', 'page', 'clicks', 'impressions']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading GSC file: {str(e)}")
    
    @staticmethod
    def load_similarity(file) -> pd.DataFrame:
        """Load semantic similarity report"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Map to standard column names
            if len(df.columns) >= 3:
                df = df.iloc[:, :3]  # Take first 3 columns
                df.columns = ['primary_url', 'secondary_url', 'semantic_similarity']
            else:
                raise ValueError("Similarity file needs at least 3 columns")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading similarity file: {str(e)}")
