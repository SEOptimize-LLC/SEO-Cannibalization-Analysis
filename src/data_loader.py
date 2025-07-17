"""
Data loading and validation module - handles exact file formats
"""

import pandas as pd


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
                'Query': 'query', 'Search Query': 'query',
                'Page': 'page', 'URL': 'page', 'url': 'page',
                'Clicks': 'clicks', 'clicks': 'clicks',
                'Impressions': 'impressions', 'impressions': 'impressions',
                'Position': 'position', 'Avg. Position': 'position'
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
        """Load semantic similarity report with flexible column handling"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Handle exact column names from your file
            if 'Address' in df.columns and 'Closest Semantically Similar Address' in df.columns:
                # Exact column names from your file
                df = df[['Address', 'Closest Semantically Similar Address',
                         'Semantic Similarity Score']]
                df.columns = ['primary_url', 'secondary_url', 'semantic_similarity']
            elif len(df.columns) >= 3:
                # Take first 3 columns and map them
                df = df.iloc[:, :3]
                df.columns = ['primary_url', 'secondary_url', 'semantic_similarity']
            else:
                raise ValueError("File must have at least 3 columns")
            
            # Ensure semantic_similarity is numeric
            df['semantic_similarity'] = pd.to_numeric(
                df['semantic_similarity'], errors='coerce')
            df = df.dropna(subset=['semantic_similarity'])
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading similarity file: {str(e)}")
