"""
SEO Cannibalization Analysis
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loaders.gsc_loader import GSCLoader
from src.data_loaders.similarity_loader import SimilarityLoader
from src.processors.url_filter import URLFilter
from src.analyzers.action_classifier import ActionClassifier
from src.analyzers.priority_assigner import PriorityAssigner
from src.formatters.report_formatter import ReportFormatter


class StreamlitSEOAnalyzer:
    """Streamlit-compatible SEO analyzer"""
    
    def __init__(self):
        self.gsc_loader = GSCLoader()
        self.similarity_loader = SimilarityLoader()
        self.url_filter = URLFilter()
        self.action_classifier = ActionClassifier()
        self.priority_assigner = PriorityAssigner()
        self.report_formatter = ReportFormatter()
    
    def run_analysis(self, gsc_df, similarity_df):
        """Run analysis with DataFrames instead of file paths"""
        try:
            # Step 1: Filter URLs
            gsc_df = self.url_filter.filter_dataframe(gsc_df)
            
            # Step 2: Aggregate GSC metrics
            gsc_metrics = self.gsc_loader.aggregate_metrics(gsc_df)
            
            # Step 3: Validate URLs against GSC data
            gsc_urls = set(gsc_metrics['page'])
            similarity_df = self.similarity_loader.validate_urls_against_gsc(
                similarity_df, gsc_urls
            )
            
            # Step 4: Merge data
            merged_df = self._merge_data(similarity_df, gsc_metrics)
            
            # Step 5: Calculate total clicks for priority assignment
            total_clicks = gsc_metrics['total_clicks'].sum()
            
            # Step 6: Classify actions and assign priorities
            merged_df['recommended_action'] = merged_df.apply(
                self.action_classifier.classify, axis=1
            )
            merged_df['priority'] = merged_df.apply(
                lambda row: self.priority_assigner.assign(row, total_clicks), axis=1
            )
            
            # Step 7: Format final report
            final_report = self.report_formatter.format_report(merged_df)
            
            return final_report
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return None
    
    def _merge_data(self, similarity_df, gsc_metrics):
        """Merge similarity data with GSC metrics"""
        # Rename columns for consistency
        gsc_metrics = gsc_metrics.rename(columns={
            'page': 'url',
            'indexed_queries': 'indexed_queries',
            'total_clicks': 'clicks',
            'total_impressions': 'impressions'
        })
        
        # Merge primary URL data
        merged = similarity_df.merge(
            gsc_metrics,
            left_on='primary_url',
            right_on='url',
            how='left',
            suffixes=('', '_primary')
        )
        
        # Rename primary columns
        merged = merged.rename(columns={
            'indexed_queries': 'primary_url_indexed_queries',
            'clicks': 'primary_url_clicks',
            'impressions': 'primary_url_impressions'
        })
        
        # Merge secondary URL data
        merged = merged.merge(
            gsc_metrics,
            left_on='secondary_url',
            right_on='url',
            how='left',
            suffixes=('', '_secondary')
        )
        
        # Rename secondary columns
        merged = merged.rename(columns={
            'indexed_queries': 'secondary_url_indexed_queries',
            'clicks': 'secondary_url_clicks',
            'impressions': 'secondary_url_impressions'
        })
        
        # Clean up temporary columns
        columns_to_drop = [col for col in merged.columns 
                          if col.endswith(('_primary', '_secondary'))]
        merged = merged.drop(columns=columns_to_drop)
        merged = merged.drop(columns=['url'], errors='ignore')
        
        # Fill NaN values with 0 for numeric columns
        numeric_cols = [
            'primary_url_indexed_queries', 'primary_url_clicks',
            'primary_url_impressions',
            'secondary_url_indexed_queries', 'secondary_url_clicks',
            'secondary_url_impressions'
        ]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
        
        return merged


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="SEO Cannibalization Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 SEO Cannibalization Analysis Tool")
    st.markdown("Analyze keyword cannibalization using GSC data and semantic similarity")
    
    # Initialize analyzer
    analyzer = StreamlitSEOAnalyzer()
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # GSC Data Upload
        st.subheader("Google Search Console Data")
        gsc_file = st.file_uploader(
            "Upload GSC CSV/Excel file",
            type=['csv', 'xlsx'],
            help="Upload your Google Search Console export"
        )
        
        # Similarity Data Upload
        st.subheader("Semantic Similarity Data")
        similarity_file = st.file_uploader(
            "Upload Similarity CSV/Excel file",
            type=['csv', 'xlsx'],
            help="Upload semantic similarity scores between URLs"
        )
        
        # Analysis button
        analyze_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if gsc_file and similarity_file and analyze_button:
        with st.spinner("🔄 Processing data..."):
            try:
                # Load GSC data
                if gsc_file.name.endswith('.csv'):
                    gsc_df = pd.read_csv(gsc_file)
                else:
                    gsc_df = pd.read_excel(gsc_file)
                
                # Load similarity data
                if similarity_file.name.endswith('.csv'):
                    similarity_df = pd.read_csv(similarity_file)
                else:
                    similarity_df = pd.read_excel(similarity_file)
                
                # Run analysis
                results = analyzer.run_analysis(gsc_df, similarity_df)
                
                if results is not None and not results.empty:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Summary statistics
                    st.header("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total URL Pairs", len(results))
                    with col2:
                        actions = results['recommended_action'].value_counts()
                        st.metric("Actions Needed", len(actions[actions > 0]))
                    with col3:
                        high_priority = len(results[results['priority'] == 'High'])
                        st.metric("High Priority", high_priority)
                    
                    # Action distribution
                    st.header("📋 Action Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("By Action Type")
                        action_counts = results['recommended_action'].value_counts()
                        st.bar_chart(action_counts)
                    with col2:
                        st.subheader("By Priority")
                        priority_counts = results['priority'].value_counts()
                        st.bar_chart(priority_counts)
                    
                    # Detailed results
                    st.header("🔍 Detailed Results")
                    
                    # Filters
                    col1, col2 = st.columns(2)
                    with col1:
                        action_filter = st.multiselect(
                            "Filter by Action",
                            options=results['recommended_action'].unique(),
                            default=results['recommended_action'].unique()
                        )
                    with col2:
                        priority_filter = st.multiselect(
                            "Filter by Priority",
                            options=results['priority'].unique(),
                            default=results['priority'].unique()
                        )
                    
                    # Apply filters
                    filtered_results = results[
                        (results['recommended_action'].isin(action_filter)) &
                        (results['priority'].isin(priority_filter))
                    ]
                    
                    # Display results
                    st.dataframe(filtered_results, use_container_width=True)
                    
                    # Download button
                    csv = filtered_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="cannibalization_analysis.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No results generated. Please check your data.")
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.info("Please check your file formats and column names.")
    
    elif analyze_button and (not gsc_file or not similarity_file):
        st.warning("Please upload both GSC and similarity data files.")
    
    else:
        st.info("👈 Please upload your data files using the sidebar to begin analysis")


if __name__ == "__main__":
    main()
