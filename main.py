"""
main.py - SEO Cannibalization Analysis Tool
Enterprise-grade Streamlit application for analyzing content cannibalization
using semantic similarity and GSC data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import logging
import traceback
from pathlib import Path

# Import your existing modules
from src.utils.config import Config
from src.processors.url_cleaner import URLCleaner
from src.data_loaders.gsc_loader import GSCLoader
from src.data_loaders.similarity_loader import SimilarityLoader  
from src.processors.data_aggregator import DataAggregator
from src.analyzers.cannibalization_analyzer import CannibalizationAnalyzer
from src.analyzers.priority_calculator import PriorityCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-msg {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .warning-msg {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'gsc_data': None,
        'similarity_data': None,
        'analysis_results': None,
        'config': None,
        'analysis_complete': False,
        'step': 1
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_config():
    """Load application configuration"""
    try:
        if st.session_state.config is None:
            st.session_state.config = Config()
        return st.session_state.config
    except Exception as e:
        return None

def handle_gsc_upload(uploaded_file):
    """Handle GSC data upload and validation"""
    try:
        # Use your GSCLoader class
        gsc_loader = GSCLoader()
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and validate data
        gsc_data = gsc_loader.load(temp_path)
        
        # Clean up temp file
        temp_path.unlink()
        
        # Store in session state
        st.session_state.gsc_data = gsc_data
        
        return True, gsc_data
        
    except Exception as e:
        logger.error(f"Error loading GSC data: {str(e)}")
        return False, str(e)

def handle_similarity_upload(uploaded_file):
    """Handle similarity data upload and validation"""
    try:
        # Use your SimilarityLoader class
        similarity_loader = SimilarityLoader()
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and validate data
        similarity_data = similarity_loader.load(temp_path)
        
        # Clean up temp file
        temp_path.unlink()
        
        # Store in session state
        st.session_state.similarity_data = similarity_data
        
        return True, similarity_data
        
    except Exception as e:
        logger.error(f"Error loading similarity data: {str(e)}")
        return False, str(e)

def run_analysis():
    """Run the complete cannibalization analysis using your existing classes"""
    try:
        config = st.session_state.config
        gsc_data = st.session_state.gsc_data
        similarity_data = st.session_state.similarity_data
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # DEBUG: Initial data check
        st.write("🔍 **PIPELINE DIAGNOSTICS:**")
        st.write(f"✅ GSC Data: {len(gsc_data)} rows, {gsc_data['page'].nunique()} unique pages")
        st.write(f"✅ Similarity Data: {len(similarity_data)} rows")
        
        # Show sample URLs for format comparison
        st.write("📋 **URL Format Check:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**GSC Sample URLs:**")
            for url in gsc_data['page'].head(3):
                st.write(f"- `{url}`")
        with col2:
            st.write("**Similarity Sample URLs:**")
            for url in similarity_data.iloc[:3, 0]:  # First URL column
                st.write(f"- `{url}`")
        
        # Step 1: Clean URLs (10%)
        status_text.text("Cleaning and filtering URLs...")
        progress_bar.progress(10)
        
        url_cleaner = URLCleaner(config)
        gsc_cleaned = url_cleaner.clean_dataframe(gsc_data, ['page'])
        similarity_cleaned = url_cleaner.clean_dataframe(
            similarity_data, 
            ['primary_url', 'secondary_url'] if 'primary_url' in similarity_data.columns 
            else [similarity_data.columns[0], similarity_data.columns[1]]
        )
        
        # DEBUG: After URL cleaning
        st.write(f"🧹 After URL cleaning: GSC={len(gsc_cleaned)}, Similarity={len(similarity_cleaned)}")
        if len(gsc_cleaned) == 0:
            st.error("❌ All GSC data was filtered out during URL cleaning!")
            return False, "URL cleaning removed all GSC data"
        
        # Step 2: Aggregate GSC data (30%)
        status_text.text("Aggregating GSC data by URL...")
        progress_bar.progress(30)
        
        aggregator = DataAggregator()
        gsc_aggregated = aggregator.aggregate_gsc_data(gsc_cleaned)
        
        # DEBUG: After aggregation
        st.write(f"📊 After aggregation: {len(gsc_aggregated) if gsc_aggregated is not None else 0} rows")
        
        # Step 3: Merge with similarity data (50%)
        status_text.text("Merging GSC and similarity data...")
        progress_bar.progress(50)
        
        merged_data = aggregator.merge_with_similarity(similarity_cleaned)
        
        # DEBUG: Critical merge check
        st.write(f"🔗 After merge: {len(merged_data) if merged_data is not None else 0} rows")
        if merged_data is None or len(merged_data) == 0:
            st.error("❌ ZERO RESULTS AFTER MERGE - This is your problem!")
            
            # Debug URL matching
            if hasattr(aggregator, 'gsc_aggregated') and aggregator.gsc_aggregated is not None:
                gsc_urls = set(aggregator.gsc_aggregated.iloc[:, 0] if len(aggregator.gsc_aggregated.columns) > 0 else [])
                sim_urls = set(similarity_cleaned.iloc[:, 0]) | set(similarity_cleaned.iloc[:, 1])
                matches = len(gsc_urls.intersection(sim_urls))
                st.write(f"🔍 URL overlap check: {matches} URLs match between datasets")
                if matches == 0:
                    st.write("**Root cause: No URL matches between GSC and similarity data**")
            
            return False, "No data after merge - URL format mismatch likely"
        
        # Step 4: Analyze cannibalization patterns (70%)
        status_text.text("Analyzing cannibalization patterns...")
        progress_bar.progress(70)
        
        analyzer = CannibalizationAnalyzer(config)
        analyzed_data = analyzer.analyze(merged_data)
        
        # DEBUG: After analysis
        st.write(f"🔬 After analysis: {len(analyzed_data) if analyzed_data is not None else 0} rows")
        
        # Step 5: Calculate priorities (90%)
        status_text.text("Calculating priority levels...")
        progress_bar.progress(90)
        
        priority_calc = PriorityCalculator(config)
        final_results = priority_calc.calculate_priorities(analyzed_data)
        
        # DEBUG: Final results
        st.write(f"🎯 Final results: {len(final_results) if final_results is not None else 0} rows")
        
        # Complete (100%)
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.analysis_results = final_results
        st.session_state.analysis_complete = True
        
        return True, final_results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        st.exception(e)  # Show full traceback in Streamlit
        return False, str(e)

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Load configuration
    config = load_config()
    if config is None:
        st.error("❌ Failed to load configuration. Please ensure config.yaml exists and is properly formatted.")
        st.stop()
    
    # Title and header
    st.title("🔍 SEO Cannibalization Analyzer")
    st.markdown("### Enterprise-Grade Content Cannibalization Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display current config summary
        st.subheader("Current Settings")
        if config:
            st.write(f"**Similarity Thresholds:**")
            st.write(f"- High: {config.similarity_thresholds.get('high', 0.90)}")
            st.write(f"- Medium: {config.similarity_thresholds.get('medium', 0.89)}")
            
            st.write(f"**URL Filters:**")
            st.write(f"- Excluded Parameters: {len(config.excluded_parameters)}")
            st.write(f"- Excluded Pages: {len(config.excluded_pages)}")
            st.write(f"- Excluded Patterns: {len(config.excluded_patterns)}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Upload", 
        "🔍 Analysis", 
        "📈 Results", 
        "📥 Export"
    ])
    
    # Tab 1: Data Upload
    with tab1:
        st.header("Data Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Google Search Console Data")
            gsc_file = st.file_uploader(
                "Upload GSC Export",
                type=['csv', 'xlsx'],
                help="CSV or Excel file with columns: query, page, clicks, impressions, position",
                key="gsc_upload"
            )
            
            if gsc_file:
                success, result = handle_gsc_upload(gsc_file)
                if success:
                    st.markdown('<div class="success-msg">✅ GSC data loaded successfully!</div>', 
                               unsafe_allow_html=True)
                    st.write(f"**Rows:** {len(result):,}")
                    st.write(f"**Unique Pages:** {result['page'].nunique():,}")
                    st.write(f"**Unique Queries:** {result['query'].nunique():,}")
                    st.write(f"**Total Clicks:** {int(result['clicks'].sum()):,}")
                    
                    with st.expander("Preview GSC Data"):
                        st.dataframe(result.head())
                else:
                    st.markdown(f'<div class="error-msg">❌ Error loading GSC data: {result}</div>', 
                               unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔗 Semantic Similarity Data")
            similarity_file = st.file_uploader(
                "Upload Similarity Data",
                type=['csv'],
                help="CSV file with columns: primary_url, secondary_url, similarity_score",
                key="similarity_upload"
            )
            
            if similarity_file:
                success, result = handle_similarity_upload(similarity_file)
                if success:
                    st.markdown('<div class="success-msg">✅ Similarity data loaded successfully!</div>', 
                               unsafe_allow_html=True)
                    st.write(f"**URL Pairs:** {len(result):,}")
                    st.write(f"**Unique URLs:** {pd.concat([result['primary_url'], result['secondary_url']]).nunique():,}")
                    st.write(f"**Avg Similarity:** {result['similarity_score'].mean():.3f}")
                    
                    with st.expander("Preview Similarity Data"):
                        st.dataframe(result.head())
                else:
                    st.markdown(f'<div class="error-msg">❌ Error loading similarity data: {result}</div>', 
                               unsafe_allow_html=True)
    
    # Tab 2: Analysis
    with tab2:
        st.header("Cannibalization Analysis")
        
        if st.session_state.gsc_data is None or st.session_state.similarity_data is None:
            st.markdown('<div class="warning-msg">⚠️ Please upload both GSC and Similarity data before running analysis.</div>', 
                       unsafe_allow_html=True)
        else:
            st.success("✅ All required data loaded. Ready for analysis!")
            
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                success, result = run_analysis()
                
                if success:
                    st.success("🎉 Analysis completed successfully!")
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total URL Pairs", f"{len(result):,}")
                    
                    with col2:
                        actionable = result[~result['recommended_action'].isin(['Remove', 'False Positive'])]
                        st.metric("Actionable Pairs", f"{len(actionable):,}")
                    
                    with col3:
                        high_priority = result[result['priority'] == 'High']
                        st.metric("High Priority", f"{len(high_priority):,}")
                    
                    with col4:
                        merge_actions = result[result['recommended_action'] == 'Merge']
                        st.metric("Merge Opportunities", f"{len(merge_actions):,}")
                        
                else:
                    st.markdown(f'<div class="error-msg">❌ Analysis failed: {result}</div>', 
                               unsafe_allow_html=True)
    
    # Tab 3: Results
    with tab3:
        st.header("Analysis Results")
        
        if not st.session_state.analysis_complete:
            st.info("🔄 Run analysis first to view results.")
        else:
            results = st.session_state.analysis_results
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                action_filter = st.selectbox(
                    "Filter by Action",
                    options=['All'] + list(results['recommended_action'].unique()),
                    key="action_filter"
                )
            
            with col2:
                priority_filter = st.selectbox(
                    "Filter by Priority",
                    options=['All'] + list(results['priority'].dropna().unique()),
                    key="priority_filter"
                )
            
            with col3:
                min_similarity = st.slider(
                    "Minimum Similarity Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                    key="similarity_filter"
                )
            
            # Apply filters
            filtered_results = results.copy()
            
            if action_filter != 'All':
                filtered_results = filtered_results[filtered_results['recommended_action'] == action_filter]
            
            if priority_filter != 'All':
                filtered_results = filtered_results[filtered_results['priority'] == priority_filter]
            
            filtered_results = filtered_results[filtered_results['similarity_score'] >= min_similarity]
            
            st.write(f"**Showing {len(filtered_results):,} of {len(results):,} URL pairs**")
            
            # Display results
            if len(filtered_results) > 0:
                st.dataframe(
                    filtered_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "similarity_score": st.column_config.NumberColumn(
                            "Similarity Score",
                            format="%.3f"
                        ),
                        "primary_url_clicks": st.column_config.NumberColumn(
                            "Primary URL Clicks",
                            format="%d"
                        ),
                        "secondary_url_clicks": st.column_config.NumberColumn(
                            "Secondary URL Clicks", 
                            format="%d"
                        )
                    }
                )
            else:
                st.info("No results match the current filters.")
    
    # Tab 4: Export
    with tab4:
        st.header("Export Results")
        
        if not st.session_state.analysis_complete:
            st.info("🔄 Complete analysis first to enable exports.")
        else:
            results = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Full Results")
                
                # Create Excel file
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results.to_excel(writer, sheet_name='All_Results', index=False)
                    
                    # Separate sheets for different actions
                    for action in results['recommended_action'].unique():
                        action_data = results[results['recommended_action'] == action]
                        sheet_name = action.replace(' ', '_')[:31]  # Excel sheet name limit
                        action_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Full Analysis (Excel)",
                    data=buffer,
                    file_name=f"cannibalization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                st.subheader("🎯 Action Items")
                
                # Create action items CSV
                actionable = results[~results['recommended_action'].isin(['Remove', 'False Positive'])]
                actionable_sorted = actionable.sort_values(['priority', 'similarity_score'], 
                                                         ascending=[False, False])
                
                csv = actionable_sorted.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Action Items (CSV)",
                    data=csv,
                    file_name=f"action_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.write(f"**Actionable Items:** {len(actionable):,}")
                if len(actionable) > 0:
                    st.write("**Priority Distribution:**")
                    priority_counts = actionable['priority'].value_counts()
                    for priority, count in priority_counts.items():
                        st.write(f"- {priority}: {count}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>SEO Cannibalization Analyzer | Enterprise Edition</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
