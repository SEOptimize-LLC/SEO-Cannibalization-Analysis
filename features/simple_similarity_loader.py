"""
Simple Semantic Similarity Loader
Handles semantic similarity data loading and scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SimpleSemanticSimilarityLoader:
    """Simple loader for semantic similarity data"""
    
    def __init__(self):
        self.similarity_data = {}
        self.url_mapping = {}
    
    def load_similarity_data(self, file_path: str) -> bool:
        """
        Load semantic similarity data from CSV
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            df = pd.read_csv(file_path)
            
            # Handle different column naming conventions
            if 'Address' in df.columns and 'Closest Semantically Similar Address' in df.columns:
                # Standard format
                for _, row in df.iterrows():
                    url1 = str(row['Address'])
                    url2 = str(row['Closest Semantically Similar Address'])
                    similarity = float(row.get('Semantic Similarity Score', 0.5))
                    
                    # Store bidirectional mapping
                    if url1 not in self.similarity_data:
                        self.similarity_data[url1] = {}
                    if url2 not in self.similarity_data:
                        self.similarity_data[url2] = {}
                    
                    self.similarity_data[url1][url2] = similarity
                    self.similarity_data[url2][url1] = similarity
                    
            elif 'url1' in df.columns and 'url2' in df.columns:
                # Alternative format
                for _, row in df.iterrows():
                    url1 = str(row['url1'])
                    url2 = str(row['url2'])
                    similarity = float(row.get('similarity', 0.5))
                    
                    if url1 not in self.similarity_data:
                        self.similarity_data[url1] = {}
                    if url2 not in self.similarity_data:
                        self.similarity_data[url2] = {}
                    
                    self.similarity_data[url1][url2] = similarity
                    self.similarity_data[url2][url1] = similarity
            
            else:
                # Try to find URL columns
                url_cols = [col for col in df.columns if 'url' in col.lower()]
                if len(url_cols) >= 2:
                    url1_col, url2_col = url_cols[:2]
                    similarity_col = None
                    
                    for col in df.columns:
                        if 'similarity' in col.lower():
                            similarity_col = col
                            break
                    
                    if similarity_col is None:
                        similarity_col = df.columns[-1]  # Last column as similarity
                    
                    for _, row in df.iterrows():
                        url1 = str(row[url1_col])
                        url2 = str(row[url2_col])
                        similarity = float(row[similarity_col])
                        
                        if url1 not in self.similarity_data:
                            self.similarity_data[url1] = {}
                        if url2 not in self.similarity_data:
                            self.similarity_data[url2] = {}
                        
                        self.similarity_data[url1][url2] = similarity
                        self.similarity_data[url2][url1] = similarity
            
            return True
            
        except Exception as e:
            print(f"Error loading similarity data: {e}")
            return False
    
    def get_similarity_score(self, url1: str, url2: str) -> float:
        """
        Get similarity score between two URLs
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            Similarity score (0-1), defaults to 0.5 if not found
        """
        if url1 in self.similarity_data and url2 in self.similarity_data[url1]:
            return self.similarity_data[url1][url2]
        
        # Fallback to basic similarity calculation
        return self._calculate_basic_similarity(url1, url2)
    
    def _calculate_basic_similarity(self, url1: str, url2: str) -> float:
        """
        Calculate basic similarity based on URL structure
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            Basic similarity score (0-1)
        """
        # Simple similarity based on URL path overlap
        path1 = url1.split('://')[-1].split('?')[0]
        path2 = url2.split('://')[-1].split('?')[0]
        
        # Split into components
        parts1 = path1.lower().split('/')
        parts2 = path2.lower().split('/')
        
        # Calculate Jaccard similarity
        set1 = set(parts1)
        set2 = set(parts2)
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity for exact domain matches
        domain1 = url1.split('://')[-1].split('/')[0]
        domain2 = url2.split('://')[-1].split('/')[0]
        
        if domain1 == domain2:
            similarity = min(similarity + 0.2, 1.0)
        
        return similarity
    
    def has_data(self) -> bool:
        """Check if similarity data has been loaded"""
        return len(self.similarity_data) > 0
    
    def get_available_urls(self) -> list:
        """Get list of URLs with similarity data"""
        return list(self.similarity_data.keys())
