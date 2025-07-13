import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Import the column mapper
from column_mapper import normalize_column_names, validate_required_columns

# Since you have helpers.py at root level, we can import directly from it
import helpers

# Page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
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

# Title and description
st.title("🔍 SEO Cannibalization Analyzer")
st.markdown("### Advanced Detection & Resolution Tool")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Threshold Settings
    st.subheader("Detection Thresholds")
    click_threshold = st.slider(
        "Minimum Click % Threshold",
        min_value=1,
        max_value=20,
        value=5,
        help="Pages with click % below this threshold won't be considered for cannibalization"
    ) / 100
    
    min_clicks = st.number_input(
        "Minimum Total Clicks",
        min_value=1,
        value=10,
        help="Queries with fewer clicks will be excluded"
    )
    
    # Brand Terms
    st.subheader("Brand Exclusions")
    brand_terms = st.text_area(
        "Brand Terms (one per line)",
        help="Enter brand terms to exclude from analysis"
    )
    brand_list = [term.strip() for term in brand_terms.split('\n') if term.strip()]
    
    # Advanced Options
    st.subheader("Advanced Analysis")
    enable_intent_detection = st.checkbox("Enable Intent Mismatch Detection", value=True)
    enable_serp_analysis = st.checkbox("Enable SERP Feature Analysis", value=True)
    enable_content_gap = st.checkbox("Enable Content Gap Analysis", value=True)
    enable_ml_insights = st.checkbox("Enable AI-Powered Insights", value=True)

# Main Content Area
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Input", "🔍 Analysis", "📈 Visualizations", "🎯 Recommendations", "📥 Export"])

# Tab 1: Data Input
with tab1:
    st.header("Data Source Selection")
    
    st.subheader("📁 CSV Upload")
    uploaded_file = st.file_uploader(
        "Upload GSC Export",
        type=['csv'],
        help="CSV should contain: page, query, clicks, impressions, position (column names will be automatically mapped)"
    )
    
    if uploaded_file:
        try:
            # Load the CSV
            df = pd.read_csv(uploaded_file)
            
            # Show original column structure
            st.subheader("Original Data Structure")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Columns:**")
                st.write(df.columns.tolist())
            with col2:
                st.write("**Data Shape:**")
                st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
            
            # Validate and normalize columns
            is_valid, missing_cols, mappings, df_normalized = validate_required_columns(df, show_mappings=True)
            
            if is_valid:
                st.session_state.df = df_normalized
                st.session_state.data_loaded = True
                
                # Show successful mapping
                st.markdown('<div class="success-msg">✅ Data loaded successfully! Column mappings applied.</div>', unsafe_allow_html=True)
                
                # Display column mappings
                if mappings:
                    st.subheader("Column Mappings Applied")
                    mapping_df = pd.DataFrame([
                        {"Original Column": original, "Mapped To": mapped}
                        for original, mapped in mappings.items()
                    ])
                    st.dataframe(mapping_df, use_container_width=True, hide_index=True)
                
                # Data preview
                with st.expander("Preview Normalized Data"):
                    st.dataframe(df_normalized.head())
                
                # Show basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Pages", f"{df_normalized['page'].nunique():,}")
                with col2:
                    st.metric("Unique Queries", f"{df_normalized['query'].nunique():,}")
                with col3:
                    st.metric("Total Clicks", f"{df_normalized['clicks'].sum():,}")
                    
            else:
                st.error(f"❌ Missing required columns after mapping: {', '.join(missing_cols)}")
                
                # Show what columns we found and what we're looking for
                st.subheader("Column Mapping Diagnostics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Available Columns:**")
                    for col in df.columns:
                        st.write(f"- {col}")
                
                with col2:
                    st.write("**Required Columns:**")
                    required_cols = ['page', 'query', 'clicks', 'impressions', 'position']
                    for col in required_cols:
                        if col in missing_cols:
                            st.write(f"- ❌ {col}")
                        else:
                            st.write(f"- ✅ {col}")
                
                # Show mapping suggestions
                if mappings:
                    st.subheader("Partial Mappings Found")
                    st.write("These columns were successfully mapped:")
                    for original, mapped in mappings.items():
                        st.write(f"- {original} → {mapped}")
                
                st.markdown("""
                **Troubleshooting Tips:**
                1. Ensure your CSV contains data for pages, search queries, clicks, impressions, and position
                2. Column names can be in various formats (e.g., "Landing Page" for page, "Avg. Pos" for position)
                3. Check for any typos in column headers
                4. Make sure numeric columns (clicks, impressions, position) contain valid numbers
                """)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.write("Please ensure your CSV file is properly formatted and try again.")

# Tab 2: Analysis
with tab2:
    st.header("Cannibalization Analysis")
    
    if st.session_state.data_loaded:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing cannibalization patterns..."):
                df = st.session_state.df
                progress_bar = st.progress(0)
                
                # Step 1: Remove brand queries
                progress_bar.progress(20)
                st.text("Filtering brand queries...")
                non_brand_df = helpers.remove_brand_queries(df, brand_list) if brand_list else df
                
                # Step 2: Calculate metrics
                progress_bar.progress(30)
                st.text("Calculating page-query metrics...")
                query_page_counts = helpers.calculate_query_page_metrics(non_brand_df)
                
                # Step 3: Filter queries
                progress_bar.progress(40)
                st.text("Filtering queries by criteria...")
                query_counts = helpers.filter_queries_by_clicks_and_pages(query_page_counts)
                
                # Step 4: Merge and aggregate
                progress_bar.progress(50)
                st.text("Merging and aggregating data...")
                wip_df = helpers.merge_and_aggregate(query_page_counts, query_counts)
                
                # Step 5: Calculate percentages
                progress_bar.progress(60)
                st.text("Calculating click percentages...")
                wip_df = helpers.calculate_click_percentage(wip_df)
                
                # Step 6: Filter by click percentage
                progress_bar.progress(70)
                st.text("Filtering by click percentage threshold...")
                # Update the function to use custom threshold
                wip_df['clicks_pct_vs_query'] = wip_df.groupby('query')['clicks'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
                queries_to_keep = wip_df[wip_df['clicks_pct_vs_query'] >= click_threshold].groupby('query').filter(lambda x: len(x) >= 2)['query'].unique()
                wip_df = wip_df[wip_df['query'].isin(queries_to_keep)]
                
                # Step 7: Merge with page clicks
                progress_bar.progress(80)
                st.text("Merging with page-level data...")
                wip_df = helpers.merge_with_page_clicks(wip_df, df)
                
                # Step 8: Define opportunities
                progress_bar.progress(90)
                st.text("Identifying opportunities...")
                final_df = helpers.define_opportunity_levels(wip_df)
                
                # Step 9: Sort and finalize
                progress_bar.progress(100)
                st.text("Finalizing results...")
                final_df = helpers.sort_and_finalize_output(final_df)
                
                # Store results
                st.session_state.analysis_results = {
                    'all_opportunities': final_df,
                    'immediate_opportunities': helpers.immediate_opps(final_df),
                    'qa_data': helpers.create_qa_dataframe(df, final_df)
                }
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                
                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries Analyzed", f"{len(df['query'].unique()):,}")
                with col2:
                    cannibalization_issues = len(final_df['query'].unique())
                    st.metric("Cannibalization Issues", f"{cannibalization_issues:,}")
                with col3:
                    immediate_opps_df = st.session_state.analysis_results['immediate_opportunities']
                    st.metric("Immediate Opportunities", f"{len(immediate_opps_df['query'].unique()):,}")
                with col4:
                    pages_to_consolidate = len(final_df['page'].unique())
                    st.metric("Pages to Review", f"{pages_to_consolidate:,}")
        
        if st.session_state.analysis_complete:
            # Detailed Results
            st.subheader("Cannibalization Issues Detected")
            results_df = st.session_state.analysis_results['all_opportunities']
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                selected_queries = st.multiselect(
                    "Filter by Query",
                    options=results_df['query'].unique(),
                    default=None
                )
            with col2:
                comment_filter = st.selectbox(
                    "Filter by Status",
                    options=['All', 'Potential Opportunity', 'Risk'],
                    index=0
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            if selected_queries:
                filtered_df = filtered_df[filtered_df['query'].isin(selected_queries)]
            if comment_filter != 'All':
                filtered_df = filtered_df[filtered_df['comment'].str.contains(comment_filter)]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Please load data in the Data Input tab first")

# Tab 3: Visualizations
with tab3:
    st.header("Visual Analysis")
    
    if st.session_state.analysis_complete:
        results_df = st.session_state.analysis_results['all_opportunities']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cannibalization Distribution
            pages_per_query = results_df.groupby('query')['page'].count().value_counts().sort_index()
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=[f'{i} Pages' for i in pages_per_query.index],
                    y=pages_per_query.values,
                    marker_color=['#28a745' if i == 2 else '#ffc107' if i == 3 else '#dc3545' for i in pages_per_query.index]
                )
            ])
            fig_dist.update_layout(
                title="Cannibalization Distribution",
                xaxis_title="Number of Competing Pages",
                yaxis_title="Number of Queries",
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Opportunity vs Risk Distribution
            comment_counts = results_df['comment'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=comment_counts.index,
                values=comment_counts.values,
                hole=.3,
                marker_colors=['#28a745' if 'Opportunity' in x else '#dc3545' for x in comment_counts.index]
            )])
            fig_pie.update_layout(
                title="Opportunity vs Risk Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Click Distribution Analysis
        st.subheader("Click Distribution Analysis")
        
        # Get top 10 queries by total clicks
        top_queries = results_df.groupby('query')['clicks_query'].sum().nlargest(10).index
        top_queries_data = results_df[results_df['query'].isin(top_queries)]
        
        fig_clicks = px.bar(
            top_queries_data,
            x='query',
            y='clicks_query',
            color='page',
            title="Click Distribution for Top 10 Cannibalized Queries",
            labels={'clicks_query': 'Clicks', 'query': 'Query'}
        )
        fig_clicks.update_layout(height=500)
        st.plotly_chart(fig_clicks, use_container_width=True)
        
    else:
        st.info("Run analysis first to see visualizations")

# Tab 4: Recommendations
with tab4:
    st.header("Consolidation Recommendations")
    
    if st.session_state.analysis_complete:
        immediate_opps_df = st.session_state.analysis_results['immediate_opportunities']
        
        if not immediate_opps_df.empty:
            st.subheader("🎯 High Priority Consolidation Opportunities")
            st.markdown("These queries have 2+ pages marked as 'Potential Opportunity'")
            
            # Group by query and show recommendations
            for query in immediate_opps_df['query'].unique():
                with st.expander(f"Query: {query}"):
                    query_data = immediate_opps_df[immediate_opps_df['query'] == query].sort_values('clicks_query', ascending=False)
                    
                    # Primary page recommendation
                    primary_page = query_data.iloc[0]
                    st.markdown("**Recommended Primary Page:**")
                    st.info(f"🏆 {primary_page['page']}")
                    st.markdown(f"- Current Clicks: {primary_page['clicks_query']:,}")
                    st.markdown(f"- Average Position: {primary_page['avg_position']:.1f}")
                    
                    # Pages to redirect
                    st.markdown("**Pages to Redirect (301):**")
                    for _, page in query_data.iloc[1:].iterrows():
                        st.warning(f"↪️ {page['page']}")
                        st.markdown(f"  - Clicks to recover: {page['clicks_query']:,}")
                    
                    # Estimated impact
                    total_clicks = query_data['clicks_query'].sum()
                    recovery_estimate = (total_clicks - primary_page['clicks_query']) * 0.7
                    st.success(f"**Estimated Traffic Recovery:** +{int(recovery_estimate):,} clicks/month")
                    
                    # Content recommendations
                    if enable_content_gap:
                        st.markdown("**Content Preservation Checklist:**")
                        st.markdown("- [ ] Review unique content elements from each page")
                        st.markdown("- [ ] Merge valuable sections into primary page")
                        st.markdown("- [ ] Update meta title/description")
                        st.markdown("- [ ] Consolidate internal links")
                        st.markdown("- [ ] Set up 301 redirects")
        else:
            st.info("No immediate consolidation opportunities found. Review the Analysis tab for queries that may need manual review.")
    else:
        st.info("Complete analysis to see recommendations")

# Tab 5: Export
with tab5:
    st.header("Export Results")
    
    if st.session_state.analysis_complete:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel export
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # All opportunities
                st.session_state.analysis_results['all_opportunities'].to_excel(
                    writer, sheet_name='all_potential_opps', index=False
                )
                # Immediate opportunities
                st.session_state.analysis_results['immediate_opportunities'].to_excel(
                    writer, sheet_name='high_likelihood_opps', index=False
                )
                # QA data
                st.session_state.analysis_results['qa_data'].to_excel(
                    writer, sheet_name='risk_qa_data', index=False
                )
            
            buffer.seek(0)
            st.download_button(
                label="📊 Download Full Report (Excel)",
                data=buffer,
                file_name=f"seo_cannibalization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV export for redirect map
            immediate_opps = st.session_state.analysis_results['immediate_opportunities']
            if not immediate_opps.empty:
                redirect_data = []
                for query in immediate_opps['query'].unique():
                    query_data = immediate_opps[immediate_opps['query'] == query].sort_values('clicks_query', ascending=False)
                    primary_page = query_data.iloc[0]['page']
                    for _, page in query_data.iloc[1:].iterrows():
                        redirect_data.append({
                            'url_from': page['page'],
                            'url_to': primary_page,
                            'query': query,
                            'clicks_to_recover': page['clicks_query']
                        })
                
                redirect_df = pd.DataFrame(redirect_data)
                csv = redirect_df.to_csv(index=False)
                st.download_button(
                    label="🔄 Download Redirect Map (CSV)",
                    data=csv,
                    file_name=f"redirect_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No redirects to export")
        
        with col3:
            st.info("Integration exports coming soon!")
        
        # Preview export data
        st.subheader("Export Preview")
        export_tabs = st.tabs(["All Opportunities", "Immediate Opportunities", "QA Data"])
        
        with export_tabs[0]:
            st.dataframe(
                st.session_state.analysis_results['all_opportunities'].head(20),
                use_container_width=True,
                hide_index=True
            )
        
        with export_tabs[1]:
            st.dataframe(
                st.session_state.analysis_results['immediate_opportunities'].head(20),
                use_container_width=True,
                hide_index=True
            )
        
        with export_tabs[2]:
            st.dataframe(
                st.session_state.analysis_results['qa_data'].head(20),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Complete analysis to enable exports")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>SEO Cannibalization Analyzer v2.0 | Built with ❤️ for SEO professionals</p>
        <p>For support and updates, visit our <a href="https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis" target="_blank">GitHub repository</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
