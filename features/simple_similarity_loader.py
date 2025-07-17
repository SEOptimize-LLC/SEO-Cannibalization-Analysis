"""
Simple Semantic Similarity Loader
Loads semantic similarity scores from CSV files for URL consolidation analysis
"""

import pandas as pd
from typing import Dict, Tuple, Optional
import os

class SimpleSemanticSimilarityLoader:
    """Load semantic similarity scores from CSV files"""
    
    def __init__(self, similarity_file: str = None):
        self.similarity_data = {}
        self.similarity_file = similarity_file
        
    def load_similarity_data(self, file_path: str = None) -> Dict[Tuple[str, str], float]:
        """
        Load semantic similarity data from CSV file
        
        Args:
            file_path: Path to CSV file with semantic similarity data
            
        Returns:
            Dictionary mapping URL pairs to similarity scores
        """
        if file_path is None:
            file_path = self.similarity_file
            
        if not file_path or not os.path.exists(file_path):
            return {}
        
        try:
            # Try to load the CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['Address', 'Closest Semantically Similar Address', 'Semantic Similarity Score']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: CSV missing required columns. Expected: {required_columns}")
                return {}
            
            # Build similarity dictionary
            similarity_dict = {}
            for _, row in df.iterrows():
                url1 = row['Address']
                url2 = row['Closest Semantically Similar Address']
                score = float(str(row['Semantic Similarity Score']).replace(',', '.'))
                
                # Store both directions
                similarity_dict[(url1, url2)] = score
                similarity_dict[(url2, url1)] = score
            
            self.similarity_data = similarity_dict
            print(f"Loaded {len(similarity_dict)} similarity scores from {file_path}")
            return similarity_dict
            
        except Exception as e:
            print(f"Error loading similarity data: {str(e)}")
            return {}
    
    def get_similarity_score(self, url1: str, url2: str) -> float:
        """Get similarity score for a URL pair"""
        return self.similarity_data.get((url1, url2), 0.0)
    
    def get_similar_urls(self, url: str, threshold: float = 0.8) -> Dict[str, float]:
        """Get all URLs similar to the given URL above threshold"""
        similar = {}
        for (u1, u2), score in self.similarity_data.items():
            if u1 == url and score >= threshold:
                similar[u2] = score
        return similar
    
    def create_similarity_matrix(self, urls: list) -> Dict[Tuple[str, str], float]:
        """Create similarity matrix for given URLs"""
        matrix = {}
        for i, url1 in enumerate(urls):
            for url2 in urls[i+1:]:
                score = self.get_similarity_score(url1, url2)
                if score > 0:
                    matrix[(url1, url2)] = score
                    matrix[(url2, url1)] = score
        return matrix
    
    def get_average_similarity(self, url: str) -> float:
        """Get average similarity score for a URL"""
        scores = [score for (u1, u2), score in self.similarity_data.items() 
                 if u1 == url or u2 == url]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_top_similar_urls(self, url: str, top_n: int = 5) -> list:
        """Get top N most similar URLs"""
        similar = self.get_similar_urls(url, threshold=0.0)
        return sorted(similar.items(), key=lambda x: x[1], reverse=True)[:top_n]
