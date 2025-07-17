#!/usr/bin/env python3
"""
SEO Cannibalization Analysis Tool - Streamlit Interface
Version: 2.0.0
Author: SEOptimize LLC
"""

import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime
from pathlib import Path
import tempfile
import os

# Import the analysis components
from src.utils.config import Config
from src.data_loaders.gsc_loader import GSCLoader
from src.data_loaders.similarity_loader import SimilarityLoader
from src.processors.url_cleaner import URLCleaner
from src.processors.data_aggregator import DataAggregator
from src.analyzers.cannibalization_analyzer import CannibalizationAnalyzer
from src.analyzers.priority_calculator import PriorityCalculator

# Page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .priority-high {
        color: #dc2626;
        font-weight: bold;
    }
    .priority-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .priority-low {
        color: #10b981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitSEOAnalyzer:
    """Streamlit interface for SEO Cannibalization Analysis"""
    
    def __init__(self):
        self.config = Config()
        self.gsc_loader = GSCLoader()
        self.similarity_loader = SimilarityLoader()
        self.url_cleaner = URLCleaner(self.config)
        self.aggregator = DataAggregator()
        self.analyzer = CannibalizationAnalyzer(self.config)
        self.priority_calculator = PriorityCalculator(self.config)
    
    def run_analysis(self, gsc_file, similarity_file):
        """Run the analysis pipeline"""
        try:
            # Progress bar
            progress = st.progress(0)
            status = st.empty()
            
            # Step 1: Load GSC data
            status.text("Loading Google Search Console data...")
            progress.progress(15)
            
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(gsc_file.name).suffix) as tmp_gsc:
                tmp_gsc.write(gsc_file.getvalue())
                gsc_path = tmp_gsc.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(similarity_file.name).suffix) as tmp_sim:
                tmp_sim.write(similarity_file.getvalue())
                sim_path = tmp_sim.name
            
            gsc_data = self.gsc_loader.load(gsc_path)
            
            # Step 2: Load similarity data
            status.text("Loading semantic similarity data...")
            progress.progress(30)
            similarity_data = self.similarity_loader.load(sim_path)
            
            # Step 3: Clean URLs
            status.text("Cleaning and filtering URLs...")
            progress.progress(45)
            gsc_data = self.url_cleaner.clean_dataframe(gsc_data, ['page'])
            similarity_data = self.url_cleaner.clean_dataframe(
                similarity_data, 
                ['primary_url', 'secondary_url']
            )
            
            # Step 4: Aggregate GSC data
            status.text("Aggregating performance metrics...")
            progress.progress(60)
            self.aggregator.aggregate_gsc_data(gsc_data)
            
            # Step 5: Merge data
            status.text("Merging data sources...")
            progress.progress(75)
            merged_data = self.aggregator.merge_with_similarity(similarity_data)
            
            # Step 6: Analyze cannibalization
            status.text("Analyzing cannibalization patterns...")
            progress.progress(85)
            analyzed_data = self.analyzer.analyze(merged_data)
            
            # Step 7: Calculate priorities
            status.text("Calculating priorities...")
            progress.progress(95)
            final_data = self.priority_calculator.calculate_priorities(analyzed_data)
            
            # Cleanup temp files
            os.unlink(gsc_path)
            os.unlink(sim_path)
            
            progress.progress(100)
            status.text("Analysis complete!")
            
            return final_data
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None


def download_csv(df, filename):
    """Generate download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 SEO Cannibalization Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify and fix keyword cannibalization issues on your website</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Upload Files")
        st.markdown("Upload your data files to begin the analysis")
        
        gsc_file = st.file_uploader(
            "Google Search Console Report",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your GSC export with query, page, clicks, impressions, and position columns"
        )
        
        similarity_file = st.file_uploader(
            "Semantic Similarity Report",
            type=['csv'],
            help="Upload your similarity analysis CSV file"
        )
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📋 Required File Formats"):
            st.markdown("""
            **GSC Report must include:**
            - `query`: Search query
            - `page`: URL
            - `clicks`: Click count
            - `impressions`: Impression count
            - `position`: Average position
            
            **Similarity Report must include:**
            - `Address`: Primary URL
            - `Closest Semantically Similar Address`: Secondary URL
            - `Semantic Similarity Score`: Score (0-1)
            """)
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Version:** 2.0.0  
            **Author:** SEOptimize LLC  
            
            This tool helps identify when multiple pages compete for the same keywords, 
            potentially diluting your SEO performance.
            """)
    
    # Main content area
    if gsc_file and similarity_file:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            analyzer = StreamlitSEOAnalyzer()
            
            with st.spinner("Analyzing your data..."):
                results = analyzer.run_analysis(gsc_file, similarity_file)
            
            if results is not None and len(results) > 0:
                st.success("✅ Analysis completed successfully!")
                
                # Display summary metrics
                st.markdown("### 📊 Summary Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total URL Pairs", len(results))
                
                with col2:
                    high_priority = len(results[results['priority'] == 'High'])
                    st.metric("High Priority Issues", high_priority)
                
                with col3:
                    actions_needed = len(results[~results['recommended_action'].isin(['Remove', 'False Positive'])])
                    st.metric("Actions Required", actions_needed)
                
                with col4:
                    avg_similarity = results['similarity_score'].mean()
                    st.metric("Avg Similarity", f"{avg_similarity:.2%}")
                
                # Action distribution
                st.markdown("### 🎯 Recommended Actions")
                
                action_counts = results['recommended_action'].value_counts()
                
                # Create a more visual representation
                action_data = pd.DataFrame({
                    'Action': action_counts.index,
                    'Count': action_counts.values,
                    'Percentage': (action_counts.values / len(results) * 100).round(1)
                })
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(action_data, hide_index=True, use_container_width=True)
                
                with col2:
                    st.bar_chart(action_data.set_index('Action')['Count'])
                
                # Priority breakdown
                st.markdown("### 🔥 Priority Distribution")
                
                priority_data = results[results['priority'] != 'N/A']['priority'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"<div class='metric-card'><h3 class='priority-high'>High Priority</h3><h1>{priority_data.get('High', 0)}</h1></div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"<div class='metric-card'><h3 class='priority-medium'>Medium Priority</h3><h1>{priority_data.get('Medium', 0)}</h1></div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"<div class='metric-card'><h3 class='priority-low'>Low Priority</h3><h1>{priority_data.get('Low', 0)}</h1></div>", unsafe_allow_html=True)
                
                # High priority issues
                st.markdown("### 🚨 Top High Priority Issues")
                
                high_priority_issues = results[results['priority'] == 'High'].head(10)
                
                if len(high_priority_issues) > 0:
                    # Format the display
                    display_cols = ['primary_url', 'secondary_url', 'similarity_score', 
                                  'recommended_action', 'primary_url_clicks', 'secondary_url_clicks']
                    
                    display_df = high_priority_issues[display_cols].copy()
                    display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.2%}")
                    display_df.columns = ['Primary URL', 'Secondary URL', 'Similarity', 
                                        'Action', 'Primary Clicks', 'Secondary Clicks']
                    
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Primary URL": st.column_config.LinkColumn("Primary URL"),
                            "Secondary URL": st.column_config.LinkColumn("Secondary URL"),
                        }
                    )
                else:
                    st.info("No high priority issues found - your SEO health looks good!")
                
                # Full results with filters
                st.markdown("### 📋 Full Results")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
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
                
                with col3:
                    similarity_range = st.slider(
                        "Similarity Score Range",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.0, 1.0),
                        step=0.05
                    )
                
                # Apply filters
                filtered_results = results[
                    (results['recommended_action'].isin(action_filter)) &
                    (results['priority'].isin(priority_filter)) &
                    (results['similarity_score'] >= similarity_range[0]) &
                    (results['similarity_score'] <= similarity_range[1])
                ]
                
                st.info(f"Showing {len(filtered_results)} of {len(results)} results")
                
                # Display filtered results
                st.dataframe(
                    filtered_results,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
                
                # Download section
                st.markdown("### 💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'seo_cannibalization_analysis_{timestamp}.csv'
                    st.markdown(download_csv(results, filename), unsafe_allow_html=True)
                
                with col2:
                    # Create summary report
                    summary_text = f"""SEO Cannibalization Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total URL pairs analyzed: {len(results)}

Recommended Actions:
{action_counts.to_string()}

Priority Distribution:
High: {priority_data.get('High', 0)}
Medium: {priority_data.get('Medium', 0)}
Low: {priority_data.get('Low', 0)}
"""
                    st.download_button(
                        label="Download Summary Report",
                        data=summary_text,
                        file_name=f"seo_summary_{timestamp}.txt",
                        mime="text/plain"
                    )
                
    else:
        # Welcome message when no files are uploaded
        st.info("👈 Please upload both required files in the sidebar to begin the analysis")
        
        # Display sample data format
        with st.expander("📊 Sample Data Format"):
            st.markdown("**GSC Report Example:**")
            sample_gsc = pd.DataFrame({
                'query': ['seo tools', 'keyword research', 'backlink analysis'],
                'page': ['https://example.com/seo-tools', 'https://example.com/keyword-research', 'https://example.com/backlinks'],
                'clicks': [150, 230, 89],
                'impressions': [3200, 4100, 1200],
                'position': [4.5, 2.3, 8.7]
            })
            st.dataframe(sample_gsc, hide_index=True)
            
            st.markdown("**Similarity Report Example:**")
            sample_sim = pd.DataFrame({
                'Address': ['https://example.com/seo-tools', 'https://example.com/keyword-research'],
                'Closest Semantically Similar Address': ['https://example.com/seo-software', 'https://example.com/keyword-tools'],
                'Semantic Similarity Score': [0.92, 0.87]
            })
            st.dataframe(sample_sim, hide_index=True)


if __name__ == "__main__":
    main()
