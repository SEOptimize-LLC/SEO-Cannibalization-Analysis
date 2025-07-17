"""
SEO Cannibalization Analysis Tool - Main Orchestrator
Built from scratch with modular architecture
"""

import streamlit as st
import pandas as pd

from src.data_loader import DataLoader
from src.url_filter import URLFilter
from src.metrics_calculator import MetricsCalculator
from src.report_generator import ReportGenerator


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="SEO Cannibalization Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 SEO Cannibalization Analysis Tool")
    st.markdown("Built from scratch with exact specifications")
    
    with st.sidebar:
        st.header("📁 File Upload")
        
        gsc_file = st.file_uploader(
            "Upload Google Search Console Report",
            type=['csv', 'xlsx'],
            help="CSV/Excel with columns: query, page, clicks, impressions"
        )
        
        similarity_file = st.file_uploader(
            "Upload Semantic Similarity Report",
            type=['csv', 'xlsx'],
            help="CSV/Excel with columns: Address, Closest Semantically Similar Address, Semantic Similarity Score"
        )
        
        analyze_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)
    
    if gsc_file and similarity_file and analyze_button:
        with st.spinner("🔄 Processing data..."):
            try:
                # Load data
                gsc_df = DataLoader.load_gsc(gsc_file)
                similarity_df = DataLoader.load_similarity(similarity_file)
                
                st.info(f"📊 Loaded {len(gsc_df)} GSC rows and {len(similarity_df)} similarity pairs")
                
                # Filter data
                gsc_filtered = URLFilter.filter_data(gsc_df)
                st.info(f"🧹 Filtered to {len(gsc_filtered)} valid URLs")
                
                # Calculate metrics
                gsc_metrics = MetricsCalculator.calculate_metrics(gsc_filtered)
                
                # Generate report
                report = ReportGenerator.generate_report(similarity_df, gsc_metrics)
                
                if not report.empty:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Summary
                    st.header("📊 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Pairs", len(report))
                    with col2:
                        st.metric("Remove", len(report[report['recommended_action'] == 'Remove']))
                    with col3:
                        st.metric("Merge", len(report[report['recommended_action'] == 'Merge']))
                    with col4:
                        st.metric("Redirect", len(report[report['recommended_action'] == 'Redirect']))
                    
                    # Action breakdown
                    st.header("📋 Detailed Results")
                    action_counts = report['recommended_action'].value_counts()
                    st.bar_chart(action_counts)
                    
                    # Data table
                    st.dataframe(report, use_container_width=True)
                    
                    # Download
                    csv = report.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Report",
                        data=csv,
                        file_name=f"cannibalization_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No valid data after filtering")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check your file formats and try again")
    
    else:
        st.info("👈 Upload both files to begin analysis")
        
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use
    1. **Export GSC data** from Search Results → Export → CSV
    2. **Upload your semantic similarity report** from your embeddings tool
    3. **Click "Run Analysis"** to get exact results
    
    ### 🎯 Features
    - **Exact column mapping** as specified
    - **Smart URL filtering** removes unwanted pages
    - **Precise action classification** based on your rules
    - **Priority assignment** by traffic potential
    - **Downloadable CSV** with exact format
    """)


if __name__ == "__main__":
    main()
