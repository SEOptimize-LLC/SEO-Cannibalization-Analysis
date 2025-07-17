import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GSCLoader:
    """Loads and validates Google Search Console data"""
    
    REQUIRED_COLUMNS = ['query', 'page', 'clicks', 'impressions', 'position']
    
    def __init__(self):
        self.data = None
    
    def load(self, file_path):
        """Load GSC data from CSV or Excel file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GSC file not found: {file_path}")
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
                logger.info(f"Loaded Excel file with {len(self.data)} rows")
            elif file_path.suffix.lower() == '.csv':
                # Try different delimiters
                for delimiter in [',', ';', '\t']:
                    try:
                        self.data = pd.read_csv(file_path, delimiter=delimiter)
                        if len(self.data.columns) >= len(self.REQUIRED_COLUMNS):
                            logger.info(f"Loaded CSV with delimiter '{delimiter}' and {len(self.data)} rows")
                            break
                    except:
                        continue
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            self._validate_columns()
            self._clean_data()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading GSC data: {e}")
            raise
    
    def _validate_columns(self):
        """Ensure all required columns are present"""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Convert numeric columns
        numeric_cols = ['clicks', 'impressions', 'position']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Remove rows with null values in critical columns
        self.data = self.data.dropna(subset=['query', 'page'])
        
        # Ensure URLs are strings
        self.data['page'] = self.data['page'].astype(str).str.strip()
        self.data['query'] = self.data['query'].astype(str).str.strip()
        
        logger.info(f"Cleaned data: {len(self.data)} rows remaining")
