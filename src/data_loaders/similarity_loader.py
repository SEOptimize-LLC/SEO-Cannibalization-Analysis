import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimilarityLoader:
    """Loads and validates semantic similarity data"""
    
    def __init__(self):
        self.data = None
    
    def load(self, file_path):
        """Load similarity data from CSV file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Similarity file not found: {file_path}")
        
        try:
            # Try different delimiters
            for delimiter in [';', ',', '\t']:
                try:
                    self.data = pd.read_csv(file_path, delimiter=delimiter)
                    if len(self.data.columns) >= 3:
                        logger.info(f"Loaded CSV with delimiter '{delimiter}' and {len(self.data)} rows")
                        break
                except:
                    continue
            
            if self.data is None or len(self.data.columns) < 3:
                raise ValueError("File must have at least 3 columns")
            
            self._standardize_columns()
            self._clean_data()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading similarity data: {e}")
            raise
    
    def _standardize_columns(self):
        """Standardize column names"""
        # Map common column name variations
        column_mapping = {
            'Address': 'primary_url',
            'address': 'primary_url',
            'URL': 'primary_url',
            'url': 'primary_url',
            'Closest Semantically Similar Address': 'secondary_url',
            'Similar Address': 'secondary_url',
            'similar_url': 'secondary_url',
            'Semantic Similarity Score': 'similarity_score',
            'Similarity Score': 'similarity_score',
            'similarity': 'similarity_score',
            'score': 'similarity_score'
        }
        
        self.data = self.data.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required = ['primary_url', 'secondary_url', 'similarity_score']
        missing = set(required) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns after mapping: {missing}")
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Fix decimal comma issue (European format)
        if self.data['similarity_score'].dtype == object:
            self.data['similarity_score'] = self.data['similarity_score'].str.replace(',', '.')
        
        # Convert to numeric
        self.data['similarity_score'] = pd.to_numeric(self.data['similarity_score'], errors='coerce')
        
        # Ensure URLs are strings
        self.data['primary_url'] = self.data['primary_url'].astype(str).str.strip()
        self.data['secondary_url'] = self.data['secondary_url'].astype(str).str.strip()
        
        # Remove invalid rows
        self.data = self.data.dropna(subset=['primary_url', 'secondary_url', 'similarity_score'])
        
        # Keep only relevant columns
        self.data = self.data[['primary_url', 'secondary_url', 'similarity_score']]
        
        logger.info(f"Cleaned similarity data: {len(self.data)} rows remaining")
