"""
Simple Semantic Similarity Loader
Loads pre-calculated semantic similarity scores from CSV
"""

import pandas as pd
from typing import Dict, Tuple

class SimpleSimilarityLoader:
    """Load semantic similarity scores from CSV file"""
    
    @staticmethod
    def load_similarity_scores(file_path: str) -> Dict[Tuple[str, str], float]:
        """
        Load semantic similarity scores from CSV
        
        Expected format:
        Address,Closest Semantically Similar Address,Semantic Similarity Score
        
        Returns:
            Dictionary mapping (primary_url, secondary_url) pairs to similarity scores
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle your exact column names
            if 'Address' in df.columns and 'Closest Semantically Similar Address' in df.columns and 'Semantic Similarity Score' in df.columns:
                # Your exact format
                primary_col = 'Address'
                secondary_col = 'Closest Semantically Similar Address'
                score_col = 'Semantic Similarity Score'
            else:
                # Fallback to standard format
                primary_col = 'url1'
                secondary_col = 'url2'
                score_col = 'semantic_similarity'
            
            similarity_scores = {}
            for _, row in df.iterrows():
                url1 = str(row[primary_col]).strip()
                url2 = str(row[secondary_col]).strip()
                score = float(row[score_col])
                
                # Store both directions for easy lookup
                similarity_scores[(url1, url2)] = score
                similarity_scores[(url2, url1)] = score
            
            return similarity_scores
            
        except Exception as e:
            print(f"Error loading similarity scores: {e}")
            return {}
    
    @staticmethod
    def create_template() -> str:
        """Create a template CSV file matching your format"""
        template = """Address,Closest Semantically Similar Address,Semantic Similarity Score,Indexability,Indexability Status
https://example.com/page1,https://example.com/page2,85.5,,
https://example.com/page1,https://example.com/page3,42.3,,
https://example.com/page2,https://example.com/page3,67.8,,"""
        return template
