""" SEO Cannibalization Analysis Tool
Streamlined single-page application for keyword cannibalization detection """
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from column_mapper import normalize_column_names, validate_required_columns, prepare_gsc_data

# Page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .recommendation {
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        border-left: 4px solid #ddd;
    }
    .recommendation.high {
        background-color: #fff5f5;
        border-left-color: #ff4444;
    }
    .recommendation.medium {
        background-color: #fffbf0;
        border-left-color: #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# Configuration settings
DEFAULT_BRAND_VARIANTS = []
CLICK_PERCENTAGE_THRESHOLD = 0.1  # 10% threshold for cannibalization
MIN_CLICKS_THRESHOLD = 10         # Minimum clicks to consider
MIN_IMPRESSIONS_THRESHOLD = 100   # Minimum impressions to consider
EXCLUDED_HOMEPAGE = "https://www.trysnow.com"

def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'gsc_data' not in st.session_state:
        st.session_state.gsc_data = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'cannibalization_results' not in st.session_state:
        st.session_state.cannibalization_results = None
    if 'brand_variants' not in st.session_state:
        st.session_state.brand_variants = DEFAULT_BRAND_VARIANTS
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'cannibalization_summary' not in st.session_state:
        st.session_state.cannibalization_summary = None
    if 'page_opportunities' not in st.session_state:
        st.session_state.page_opportunities = None
    if 'consolidation_recommendations' not in st.session_state:
        st.session_state.consolidation_recommendations = None

def clean_gsc_data(df):
    """ Clean Google Search Console data by removing invalid entries """
    initial_rows = len(df)
    
    # Track cleaning stats
    cleaning_stats = {
        'initial_rows': initial_rows,
        'removed_name_errors': 0,
        'removed_non_urls': 0,
        'removed_non_english': 0,
        'removed_invalid_numbers': 0,
        'removed_empty': 0,
        'removed_with_parameters': 0,
        'removed_pages_subfolder': 0,
        'removed_homepage': 0
    }

    def remove_branded_keywords(df: pd.DataFrame, variants: list) -> pd.DataFrame:
        """
        Remove rows whose query contains any of the given brand name variants.
        """
        if not variants:
            return df
        pattern = "|".join([v.lower() for v in variants])
        mask = ~df["query"].str.lower().str.contains(pattern, na=False)
        return df.loc[mask].copy()

    # 1. Remove rows with #NAME? errors in query
    name_error_mask = df['query'].astype(str).str.contains(r'#NAME\?', na=False)
    cleaning_stats['removed_name_errors'] = name_error_mask.sum()
    df = df[~name_error_mask]
    
    # 2. Remove rows where page doesn't start with https://
    valid_url_mask = df['page'].astype(str).str.startswith('https://')
    cleaning_stats['removed_non_urls'] = (~valid_url_mask).sum()
    df = df[valid_url_mask]
    
    # 3. Remove URLs with special parameters (?, =) and tracking parameters
    param_mask = df['page'].str.contains(r'[?=&]', case=False, na=False)
    cleaning_stats['removed_with_parameters'] = param_mask.sum()
    df = df[~param_mask]
    
    # 4. Remove /pages/ subfolder URLs
    pages_mask = df['page'].str.contains(r'/pages/', case=False, na=False)
    cleaning_stats['removed_pages_subfolder'] = pages_mask.sum()
    df = df[~pages_mask]
    
    # 5. Remove homepage URL
    homepage_mask = df['page'] == EXCLUDED_HOMEPAGE
    cleaning_stats['removed_homepage'] = homepage_mask.sum()
    df = df[~homepage_mask]
    
    # 6. Remove rows with non-English queries
    english_mask = df['query'].astype(str).apply(lambda x: x.isascii())
    cleaning_stats['removed_non_english'] = (~english_mask).sum()
    df = df[english_mask]
    
    # 7. Remove rows with empty queries or pages
    empty_mask = (df['query'].astype(str).str.strip() == '') | (df['page'].astype(str).str.strip() == '')
    cleaning_stats['removed_empty'] = empty_mask.sum()
    df = df[~empty_mask]
    
    # 8. Ensure numeric columns are actually numeric
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    # Remove rows with invalid numeric values
    numeric_mask = df['clicks'].notna() & df['impressions'].notna()
    cleaning_stats['removed_invalid_numbers'] = (~numeric_mask).sum()
    df = df[numeric_mask]
    
    # 9. Additional cleaning: Remove obvious test/spam queries
    spam_patterns = ['test', 'asdf', 'xxx', '123', 'lorem ipsum']
    spam_mask = df['query'].astype(str).str.lower().str.contains('|'.join(spam_patterns), na=False)
    df = df[~spam_mask]
    
    # 10. Remove queries that are just numbers or single characters
    valid_query_mask = df['query'].astype(str).str.len() > 2
    df = df[valid_query_mask]
    
    # Calculate total removed
    total_removed = initial_rows - len(df)
    cleaning_stats['final_rows'] = len(df)
    cleaning_stats['total_removed'] = total_removed
    
    return df, cleaning_stats

def remove_branded_keywords(df, brand_variants):
    """Remove branded keyword variations from the dataset"""
    if not brand_variants:
        return df
    
    brand_pattern = '|'.join([variant.lower() for variant in brand_variants])
    mask = ~df['query'].str.lower().str.contains(brand_pattern, na=False)
    filtered_df = df[mask].copy()
    df = df.dropna(axis=1, how='all')
    return filtered_df

def filter_by_multiple_pages(df):
    """Filter to keep only keywords that have multiple pages ranking"""
    query_page_counts = df.groupby('query')['page'].nunique()
    multi_page_queries = query_page_counts[query_page_counts > 1].index
    return df[df['query'].isin(multi_page_queries)]

def filter_by_clicks(df):
    """Filter to queries where multiple pages receive clicks"""
    query_click_counts = df[df['clicks'] > 0].groupby('query')['page'].nunique()
    multi_click_queries = query_click_counts[query_click_counts > 1].index
    return df[df['query'].isin(multi_click_queries)]

def calculate_click_percentages(df):
    """Calculate click percentages for each page-query combination"""
    query_totals = df.groupby('query')['clicks'].sum().reset_index()
    query_totals.columns = ['query', 'total_clicks_query']
    
    page_totals = df.groupby('page')['clicks'].sum().reset_index()
    page_totals.columns = ['page', 'total_clicks_page']
    
    df = df.merge(query_totals, on='query', how='left')
    df = df.merge(page_totals, on='page', how='left')
    
    df['clicks_pct_vs_query'] = df['clicks'] / df['total_clicks_query']
    df['clicks_pct_vs_page'] = df['clicks'] / df['total_clicks_page']
    
    df['clicks_pct_vs_query'] = df['clicks_pct_vs_query'].fillna(0)
    df['clicks_pct_vs_page'] = df['clicks_pct_vs_page'].fillna(0)
    
    return df

def filter_by_click_percentage(df, threshold=CLICK_PERCENTAGE_THRESHOLD):
    """Keep queries where at least 2 pages have significant click share"""
    significant_pages = df[df['clicks_pct_vs_query'] >= threshold]
    queries_to_keep = significant_pages.groupby('query').size()
    queries_to_keep = queries_to_keep[queries_to_keep >= 2].index
    return df[df['query'].isin(queries_to_keep)]

def calculate_cannibalization_score(df):
    """Calculate cannibalization score for each query"""
    scores = []
    
    for query in df['query'].unique():
        query_data = df[df['query'] == query]
        
        num_pages = query_data['page'].nunique()
        total_clicks = query_data['clicks'].sum()
        total_impressions = query_data['impressions'].sum()
        
        if total_clicks > 0:
            click_distribution = query_data['clicks'] / total_clicks
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in click_distribution)
        else:
            entropy = 0
        
        score = 0
        score += min(num_pages / 5, 1) * 0.3
        
        if num_pages > 1:
            max_entropy = np.log2(num_pages)
            score += (entropy / max_entropy if max_entropy > 0 else 0) * 0.4
        
        opportunity_score = np.log10(total_clicks + 1) + np.log10(total_impressions + 1)
        score += min(opportunity_score / 10, 1) * 0.3
        
        scores.append({
            'query': query,
            'cannibalization_score': round(score, 3),
            'num_pages': num_pages,
            'total_clicks': total_clicks,
            'total_impressions': total_impressions,
            'click_entropy': round(entropy, 3)
        })
    
    return pd.DataFrame(scores)

def classify_cannibalization_severity(score):
    """Classify cannibalization severity based on score"""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

def identify_consolidation_opportunities(df, scores_df):
    """Identify pages that could be consolidated"""
    recommendations = []
    
    for query in scores_df[scores_df['cannibalization_score'] > 0.4]['query']:
        query_data = df[df['query'] == query].copy()
        query_data = query_data.sort_values('clicks', ascending=False)
        
        if len(query_data) < 2:
            continue
            
        top_pages = query_data.head(2)
        
        total_clicks = query_data['clicks'].sum()
        top_page_clicks = top_pages.iloc[0]['clicks']
        second_page_clicks = top_pages.iloc[1]['clicks']
        
        if second_page_clicks / total_clicks > 0.2:
            recommendations.append({
                'query': query,
                'primary_page': top_pages.iloc[0]['page'],
                'primary_page_clicks': top_page_clicks,
                'secondary_page': top_pages.iloc[1]['page'],
                'secondary_page_clicks': second_page_clicks,
                'total_query_clicks': total_clicks,
                'consolidation_type': 'merge' if second_page_clicks / total_clicks > 0.4 else 'redirect',
                'priority': 'High' if total_clicks > 100 else 'Medium'
            })
    
    return pd.DataFrame(recommendations)

def run_cannibalization_analysis(df, brand_variants):
    """Main analysis function"""
    df_filtered = remove_branded_keywords(df, brand_variants)
    df_filtered = filter_by_multiple_pages(df_filtered)
    df_filtered = filter_by_clicks(df_filtered)
    df_filtered = calculate_click_percentages(df_filtered)
    df_filtered = filter_by_click_percentage(df_filtered)
    
    scores_df = calculate_cannibalization_score(df_filtered)
    scores_df['severity'] = scores_df['cannibalization_score'].apply(classify_cannibalization_severity)
    
    consolidation_df = identify_consolidation_opportunities(df_filtered, scores_df)
    
    return df_filtered, scores_df, consolidation_df

def main():
    """Main application function"""
    init_session_state()
    
    st.title("🔍 SEO Cannibalization Analysis Tool")
    st.markdown("Identify and fix keyword cannibalization issues using Google Search Console data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload", "📈 Analysis Results", "📄 Page Analysis", "💡 Recommendations"])
    
    with tab1:
        st.markdown("### Upload Google Search Console Data")
        st.markdown("""
        Upload a CSV export from Google Search Console with the following columns:
        - **page** (or url, landing_page)
        - **query** (or keyword, search_query)
        - **clicks**
        - **impressions**
        - **position** (or average_position)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Select your Google Search Console export file"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                first_line = uploaded_file.readline()
                if isinstance(first_line, bytes):
                    first_line = first_line.decode('utf-8', errors='ignore')
                
                delimiter = ','
                if ';' in first_line and first_line.count(';') > first_line.count(','):
                    delimiter = ';'
                    st.info("📋 Detected semicolon (;) as delimiter")
                
                uploaded_file.seek(0)
                
                try:
                    df = pd.read_csv(uploaded_file, delimiter=delimiter, on_bad_lines='skip')
                    st.info(f"✓ Loaded file successfully")
                except Exception as e:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                
                if df is None:
                    st.error("❌ Could not parse the CSV file")
                    return
                
                original_shape = df.shape
                st.info(f"📊 Original data: {original_shape[0]:,} rows, {original_shape[1]} columns")
                
                df = normalize_column_names(df)
                
                is_valid, missing_columns = validate_required_columns(df)
                if not is_valid:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Your file has these columns: " + ", ".join(df.columns))
                    st.markdown("### 🔍 Data Sample (first 5 rows)")
                    st.dataframe(df.head(), use_container_width=True)
                    return
                
                with st.spinner("🧹 Cleaning data..."):
                    df, cleaning_stats = clean_gsc_data(df)
                
                if cleaning_stats['total_removed'] > 0:
                    st.warning(f"⚠️ Data Cleaning Results:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Removed", f"{cleaning_stats['total_removed']:,}")
                        if cleaning_stats['removed_name_errors'] > 0:
                            st.caption(f"• #NAME? errors: {cleaning_stats['removed_name_errors']}")
                        if cleaning_stats['removed_empty'] > 0:
                            st.caption(f"• Empty values: {cleaning_stats['removed_empty']}")
                    with col2:
                        if cleaning_stats['removed_non_urls'] > 0:
                            st.metric("Invalid URLs", cleaning_stats['removed_non_urls'])
                        if cleaning_stats['removed_non_english'] > 0:
                            st.caption(f"• Non-English: {cleaning_stats['removed_non_english']}")
                        if cleaning_stats['removed_with_parameters'] > 0:
                            st.metric("URLs with Parameters", cleaning_stats['removed_with_parameters'])
                        if cleaning_stats['removed_pages_subfolder'] > 0:
                            st.metric("/pages/ URLs", cleaning_stats['removed_pages_subfolder'])
                        if cleaning_stats['removed_homepage'] > 0:
                            st.metric("Homepage URLs", cleaning_stats['removed_homepage'])
                    with col3:
                        if cleaning_stats['removed_invalid_numbers'] > 0:
                            st.metric("Invalid Numbers", cleaning_stats['removed_invalid_numbers'])
                    
                    st.info(f"✅ Clean data: {cleaning_stats['final_rows']:,} rows ready for analysis")
                
                df = prepare_gsc_data(df, verbose=False)
                
                st.session_state['gsc_data'] = df
                st.session_state['data_loaded'] = True
                
                st.success(f"✅ Data loaded and cleaned successfully! {len(df):,} rows ready for analysis.")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Unique Pages", f"{df['page'].nunique():,}")
                with col3:
                    st.metric("Unique Queries", f"{df['query'].nunique():,}")
                with col4:
                    st.metric("Total Clicks", f"{df['clicks'].sum():,}")
                
                if df is not None and len(df) > 0:
                    st.markdown("#### Sample Data")
                    st.dataframe(df.head(100), use_container_width=True, height=300)
                
                    st.markdown("### 🏷️ Brand Configuration")
                    st.markdown("Enter your brand name variations to exclude from analysis (one per line)")
                
                    brand_input = st.text_area(
                        "Brand variants",
                        value="\n".join(st.session_state.brand_variants),
                        height=100,
                        help="Enter brand name variations to filter out branded searches"
                    )
                
                    if brand_input:
                        st.session_state.brand_variants = [v.strip() for v in brand_input.split('\n') if v.strip()]
                
                    if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
                        with st.spinner("Analyzing data for cannibalization issues..."):
                            processed_data, scores, recommendations = run_cannibalization_analysis(
                                df, 
                                st.session_state.brand_variants
                            )
                            
                            st.session_state.processed_data = processed_data
                            st.session_state.cannibalization_summary = scores
                            st.session_state.consolidation_recommendations = recommendations
                            st.session_state.analysis_complete = True
                            
                            st.success("✅ Analysis complete!")
                            st.balloons()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                high_count = len(scores[scores['severity'] == 'High'])
                                st.metric("High Severity Issues", high_count)
                            with col2:
                                medium_count = len(scores[scores['severity'] == 'Medium'])
                                st.metric("Medium Severity Issues", medium_count)
                            with col3:
                                total_affected = len(scores)
                                st.metric("Total Affected Queries", total_affected)
                            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format and try again.")
        else:
            st.info("👆 Please upload a Google Search Console CSV export to begin analysis.")
    
    with tab2:
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please upload data and run analysis first!")
        else:
            st.markdown("### Analysis Results")
            
            scores_df = st.session_state.cannibalization_summary
            processed_data = st.session_state.processed_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_queries = len(scores_df)
                st.metric("Total Cannibalized Queries", total_queries)
            with col2:
                high_severity = len(scores_df[scores_df['severity'] == 'High'])
                st.metric("High Severity", high_severity)
            with col3:
                total_clicks_affected = scores_df['total_clicks'].sum()
                st.metric("Total Clicks Affected", f"{total_clicks_affected:,}")
            with col4:
                avg_pages = scores_df['num_pages'].mean()
                st.metric("Avg Pages per Query", f"{avg_pages:.1f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Cannibalization Severity Distribution")
                severity_counts = scores_df['severity'].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    color_discrete_map={'High': '#ff4444', 'Medium': '#ff9800', 'Low': '#4caf50'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Top 10 Affected Queries by Clicks")
                top_queries = scores_df.nlargest(10, 'total_clicks')[['query', 'total_clicks', 'severity']]
                st.dataframe(top_queries, use_container_width=True, hide_index=True)
            
            st.markdown("### Detailed Query Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=['High', 'Medium', 'Low'],
                    default=['High', 'Medium']
                )
            with col2:
                min_clicks = st.number_input(
                    "Minimum Clicks",
                    min_value=0,
                    value=10,
                    step=10
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['cannibalization_score', 'total_clicks', 'total_impressions', 'num_pages'],
                    index=0
                )
            
            filtered_scores = scores_df[
                (scores_df['severity'].isin(severity_filter)) &
                (scores_df['total_clicks'] >= min_clicks)
            ].sort_values(sort_by, ascending=False)
            
            st.dataframe(
                filtered_scores[['query', 'severity', 'cannibalization_score', 'num_pages', 'total_clicks', 'total_impressions']],
                use_container_width=True,
                hide_index=True
            )
            
            if len(filtered_scores) > 0:
                selected_query = st.selectbox("Select a query for detailed view", filtered_scores['query'].values)
                
                if selected_query:
                    query_data = processed_data[processed_data['query'] == selected_query]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"#### Pages ranking for: {selected_query}")
                        page_summary = query_data[['page', 'clicks', 'impressions', 'position', 'clicks_pct_vs_query']].copy()
                        page_summary['clicks_pct_vs_query'] = (page_summary['clicks_pct_vs_query'] * 100).round(1)
                        page_summary.columns = ['Page', 'Clicks', 'Impressions', 'Avg Position', 'Click %']
                        st.dataframe(page_summary, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("#### Click Distribution")
                        fig = px.pie(
                            query_data, 
                            values='clicks', 
                            names='page', 
                            title=f"Click distribution for '{selected_query}'"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Export Results")
            
            export_data = scores_df.merge(
                processed_data.groupby('query').agg({
                    'page': lambda x: ' | '.join(x),
                    'clicks': 'sum',
                    'impressions': 'sum'
                }).reset_index(),
                on='query',
                how='left'
            )
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Results",
                data=csv,
                file_name=f"cannibalization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please upload data and run analysis first!")
        else:
            st.markdown("### Page-Level Cannibalization Analysis")
            
            processed_data = st.session_state.processed_data
            
            page_summary = processed_data.groupby('page').agg({
                'query': 'count',
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            
            page_summary.columns = ['Page', 'Cannibalized Queries', 'Total Clicks', 'Total Impressions']
            page_summary = page_summary.sort_values('Cannibalized Queries', ascending=False)
            
            st.markdown("#### Top 20 Pages with Most Cannibalization Issues")
            st.dataframe(page_summary.head(20), use_container_width=True, hide_index=True)
            
            selected_page = st.selectbox("Select a page for detailed view", page_summary['Page'].head(50).values)
            
            if selected_page:
                # Get all queries for this page
                page_queries = processed_data[processed_data['page'] == selected_page]
                
                # Get all other pages that compete with this page for the same queries
                competing_pages = processed_data[
                    processed_data['query'].isin(page_queries['query']) &
                    (processed_data['page'] != selected_page)
                ]
                
                st.markdown(f"#### Queries for: {selected_page}")
                query_summary = page_queries[['query', 'clicks', 'impressions', 'position', 'clicks_pct_vs_page']].copy()
                query_summary['clicks_pct_vs_page'] = (query_summary['clicks_pct_vs_page'] * 100).round(1)
                query_summary.columns = ['Query', 'Clicks', 'Impressions', 'Avg Position', 'Page Click %']
                st.dataframe(query_summary.sort_values('Clicks', ascending=False), use_container_width=True, hide_index=True)
                
                if len(competing_pages) > 0:
                    st.markdown(f"#### Pages cannibalizing with {selected_page}")
                    competing_summary = competing_pages[['page', 'query', 'clicks', 'impressions', 'position']].copy()
                    competing_summary.columns = ['Competing Page', 'Query', 'Clicks', 'Impressions', 'Avg Position']
                    st.dataframe(competing_summary.sort_values(['Query', 'Clicks'], ascending=[True, False]), use_container_width=True, hide_index=True)
    
    with tab4:
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please upload data and run analysis first!")
        else:
            st.markdown("### Consolidation Recommendations")
            
            recommendations = st.session_state.consolidation_recommendations
            
            # Export button at the top
            if len(recommendations) > 0:
                st.markdown("### Export Recommendations")
                recs_csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations",
                    data=recs_csv,
                    file_name=f"consolidation_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.markdown("---")
            
            if len(recommendations) == 0:
                st.info("No high-priority consolidation opportunities found.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_recs = len(recommendations)
                    st.metric("Total Recommendations", total_recs)
                with col2:
                    high_priority = len(recommendations[recommendations['priority'] == 'High'])
                    st.metric("High Priority", high_priority)
                with col3:
                    potential_clicks = recommendations['total_query_clicks'].sum()
                    st.metric("Total Clicks at Stake", f"{potential_clicks:,}")
                
                st.markdown("### Recommendations by Type")
                
                for rec_type in ['merge', 'redirect']:
                    type_recs = recommendations[recommendations['consolidation_type'] == rec_type]
                    
                    if len(type_recs) > 0:
                        st.markdown(f"#### {rec_type.title()} Recommendations")
                        
                        # Create expandable sections for each recommendation
                        for _, rec in type_recs.iterrows():
                            severity_class = "high" if rec['priority'] == 'High' else "medium"
                            with st.expander(f"{rec['query']} - {rec['priority']} Priority"):
                                st.markdown(f"""
                                <div class="recommendation {severity_class}">
                                    <strong>Query:</strong> {rec['query']}<br>
                                    <strong>Action:</strong> {rec['consolidation_type'].title()} 
                                    <code>{rec['secondary_page']}</code> into <code>{rec['primary_page']}</code><br>
                                    <strong>Impact:</strong> {rec['total_query_clicks']} clicks 
                                    ({rec['secondary_page_clicks']} from secondary page)<br>
                                    <strong>Priority:</strong> {rec['priority']}
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
