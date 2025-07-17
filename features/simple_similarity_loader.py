"""
Simple Semantic Similarity Loader
Loads pre-calculated semantic similarity scores between URLs
"""

import pandas as pd


class SimpleSemanticSimilarityLoader:
    """Simple semantic similarity loader for URL pairs"""
    
    def __init__(self):
        self.similarity_data = {}
    
    def load_similarity_data(self, file_path: str):
        """Load similarity data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) >= 3:
                # Expected format: url1, url2, similarity
                for _, row in df.iterrows():
                    url1, url2 = str(row.iloc[0]), str(row.iloc[1])
                    similarity = float(row.iloc[2])
                    key = tuple(sorted([url1, url2]))
                    self.similarity_data[key] = similarity
        except Exception as e:
            print(f"Could not load similarity data: {e}")
    
    def get_similarity_score(self, url1: str, url2: str) -> float:
        """Get similarity score between two URLs"""
        key = tuple(sorted([str(url1), str(url2)]))
        return self.similarity_data.get(key, 0.0)
