""" SEO Cannibalization Analysis Tool
Streamlined single-page application for keyword cannibalization detection """
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from column_mapper import normalize_column_names, validate_required_columns
from features.url_consolidation_analyzer import URLConsolidationAnalyzer

# Page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Configuration settings
DEFAULT_BRAND_VARIANTS = []
CLICK_PERCENTAGE_THRESHOLD = 0.1
MIN_CLICKS_THRESHOLD = 10
MIN_IMPRESSIONS_THRESHOLD = 100

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
    if 'url_consolidation' not in st.session_state:
        st.session_state.url_consolidation = None
    if 'embeddings_data' not in st.session_state:
        st.session_state.embeddings_data = None
    if 'cleaning_stats' not in st.session_state:
        st.session_state.cleaning_stats = None

def clean_gsc_data(df):
    """Clean Google Search Console data by removing invalid entries"""
    initial_rows = len(df)
    
    cleaning_stats = {
        'initial_rows': initial_rows,
        'removed_name_errors': 0,
        'removed_non_urls': 0,
        'removed_non_english': 0,
        'removed_invalid_numbers': 0,
        'removed_empty': 0,
        'removed_with_parameters': 0,
        'removed_pages_subfolder': 0,
        'removed_homepage': 0,
        'removed_subdomains': 0
    }

    def remove_branded_keywords(df, variants):
        if not variants:
            return df
        pattern = "|".join([v.lower() for v in variants])
        mask = ~df["query"].str.lower().str.contains(pattern, na=False)
        return df.loc[mask].copy()

    name_error_mask = df['query'].astype(str).str.contains(r'#NAME\?', na=False)
    cleaning_stats['removed_name_errors'] = name_error_mask.sum()
    df = df[~name_error_mask]
    
    valid_url_mask = df['page'].astype(str).str.startswith('https://')
    cleaning_stats['removed_non_urls'] = (~valid_url_mask).sum()
    df = df[valid_url_mask]
    
    param_mask = df['page'].str.contains(r'[?=&]', case=False, na=False)
    cleaning_stats['removed_with_parameters'] = param_mask.sum()
    df = df[~param_mask]
    
    pages_mask = df['page'].str.contains(r'/pages/', case=False, na=False)
    cleaning_stats['removed_pages_subfolder'] = pages_mask.sum()
    df = df[~pages_mask]
    
    homepage_pattern = r'^https://[^/]+/?$'
    homepage_mask = df['page'].str.contains(homepage_pattern, case=False, na=False)
    cleaning_stats['removed_homepage'] = homepage_mask.sum()
    df = df[~homepage_mask]
    
    subdomain_mask = df['page'].str.contains(r'^https://(?!www\.)([^.]+)\.[^/]+/', case=False, na=False)
    cleaning_stats['removed_subdomains'] = subdomain_mask.sum()
    df = df[~subdomain_mask]
    
    english_mask = df['query'].astype(str).apply(lambda x: x.isascii())
    cleaning_stats['removed_non_english'] = (~english_mask).sum()
    df = df[english_mask]
    
    empty_mask = (df['query'].astype(str).str.strip() == '') | (df['page'].astype(str).str.strip() == '')
    cleaning_stats['removed_empty'] = empty_mask.sum()
    df = df[~empty_mask]
    
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    numeric_mask = df['clicks'].notna() & df['impressions'].notna()
    cleaning_stats['removed_invalid_numbers'] = (~numeric_mask).sum()
    df = df[numeric_mask]
    
    spam_patterns = ['test', 'asdf', 'xxx', '123', 'lorem ipsum']
    spam_mask = df['query'].astype(str).str.lower().str.contains('|'.join(spam_patterns), na=False)
    df = df[~spam_mask]
    
    valid_query_mask = df['query'].astype(str).str.len() > 2
    df = df[valid_query_mask]
    
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

def run_cannibalization_analysis(df, brand_variants):
    """Main analysis function"""
    df_filtered = remove_branded_keywords(df, brand_variants)
    df_filtered = filter_by_multiple_pages(df_filtered)
    df_filtered = filter_by_clicks(df_filtered)
    df_filtered = calculate_click_percentages(df_filtered)
    df_filtered = filter_by_click_percentage(df_filtered)
    
    scores_df = calculate_cannibalization_score(df_filtered)
    scores_df['severity'] = scores_df['cannibalization_score'].apply(classify_cannibalization_severity)
    
    return df_filtered, scores_df

def run_url_consolidation_analysis(df, embeddings_df=None):
    """Run URL-level consolidation analysis"""
    analyzer = URLConsolidationAnalyzer()
    return analyzer.analyze_url_consolidation(df, embeddings_df)

def main():
    """Main application function"""
    init_session_state()
    
    st.title("🔍 SEO Cannibalization Analysis Tool")
    st.markdown("Identify and fix keyword cannibalization issues using Google Search Console data")
    
    with st.sidebar:
        st.markdown("### 📊 Data Upload")
        st.markdown("Upload your Google Search Console CSV export")
        
        # GSC Data Upload
        uploaded_file = st.file_uploader(
            "Choose GSC CSV file", 
            type=['csv'],
            help="Select your Google Search Console export file"
        )
        
        # Embeddings Upload (Optional)
        embeddings_file = st.file_uploader(
            "Choose embeddings CSV file (optional)",
            type=['csv'],
            help="Upload URL embeddings for enhanced semantic similarity analysis"
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
                    st.success("✓ GSC file loaded successfully")
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                
                if df is None:
                    st.error("❌ Could not parse the CSV file")
                    return
                
                original_shape = df.shape
                st.info(f"📊 Original: {original_shape[0]:,} rows, {original_shape[1]} columns")
                
                df = normalize_column_names(df)
                
                is_valid, missing_columns = validate_required_columns(df)
                if not is_valid:
                    st.error(f"❌ Missing: {', '.join(missing_columns)}")
                    st.info("Columns: " + ", ".join(df.columns))
                    return
                
                # Process embeddings file if provided
                embeddings_df = None
                if embeddings_file is not None:
                    try:
                        embeddings_df = pd.read_csv(embeddings_file)
                        st.success("✓ Embeddings file loaded successfully")
                        
                        # Show embeddings info
                        st.info(f"📊 Embeddings: {len(embeddings_df)} URLs, {len(embeddings_df.columns)-1} dimensions")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not load embeddings: {str(e)}")
                        embeddings_df = None
                
                with st.spinner("🧹 Cleaning data..."):
                    df, cleaning_stats = clean_gsc_data(df)
                
                st.session_state['gsc_data'] = df
                st.session_state['data_loaded'] = True
                st.session_state['cleaning_stats'] = cleaning_stats
                st.session_state['embeddings_data'] = embeddings_df
                
                st.markdown("### 🏷️ Brand Configuration")
                st.markdown("Enter brand name variations to exclude:")
                
                brand_input = st.text_area(
                    "Brand variants",
                    value="\n".join(st.session_state.brand_variants),
                    height=70,
                    help="One per line"
                )
                
                if brand_input:
                    st.session_state.brand_variants = [v.strip() for v in brand_input.split('\n') if v.strip()]
                
                if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🧹 Cleaning and preparing data...")
                    progress_bar.progress(10)
                    
                    processed_data, scores = run_cannibalization_analysis(
                        df, 
                        st.session_state.brand_variants
                    )
                    
                    status_text.text("🔍 Analyzing URL consolidation opportunities...")
                    progress_bar.progress(50)
                    
                    url_consolidation = run_url_consolidation_analysis(
                        df, 
                        embeddings_df
                    )
                    
                    status_text.text("📊 Finalizing results...")
                    progress_bar.progress(90)
                    
                    st.session_state.processed_data = processed_data
                    st.session_state.cannibalization_summary = scores
                    st.session_state.url_consolidation = url_consolidation
                    st.session_state.analysis_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    st.balloons()
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check your CSV format and try again.")
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📊 Dashboard", "🔗 URL Consolidation Analysis"])
    
    with tab1:
        st.markdown("### 📊 Main Dashboard")
        
        if not st.session_state.data_loaded:
            st.info("👈 Please upload data using the sidebar to begin analysis")
        else:
            df = st.session_state['gsc_data']
            cleaning_stats = st.session_state['cleaning_stats']
            
            # Data overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Unique Pages", f"{df['page'].nunique():,}")
            with col3:
                st.metric("Unique Queries", f"{df['query'].nunique():,}")
            with col4:
                st.metric("Total Clicks", f"{df['clicks'].sum():,}")
            
            # Show embeddings status
            if st.session_state.embeddings_data is not None:
                st.success("✅ Enhanced semantic analysis enabled with embeddings")
            else:
                st.info("ℹ️ Using basic semantic analysis (upload embeddings for enhanced analysis)")
            
            # Cleaning results
            if cleaning_stats and cleaning_stats['total_removed'] > 0:
                st.markdown("### 🧹 Cleaning Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows Removed", f"{cleaning_stats['total_removed']:,}")
                with col2:
                    st.metric("URLs with Parameters", cleaning_stats['removed_with_parameters'])
                with col3:
                    st.metric("/pages/ URLs", cleaning_stats['removed_pages_subfolder'])
                with col4:
                    st.metric("Subdomain URLs", cleaning_stats['removed_subdomains'])
                
                # Additional cleaning details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if cleaning_stats['removed_name_errors'] > 0:
                        st.metric("#NAME? Errors", cleaning_stats['removed_name_errors'])
                with col2:
                    if cleaning_stats['removed_non_urls'] > 0:
                        st.metric("Invalid URLs", cleaning_stats['removed_non_urls'])
                with col3:
                    if cleaning_stats['removed_non_english'] > 0:
                        st.metric("Non-English", cleaning_stats['removed_non_english'])
                with col4:
                    if cleaning_stats['removed_invalid_numbers'] > 0:
                        st.metric("Invalid Numbers", cleaning_stats['removed_invalid_numbers'])
            
            st.markdown("#### Sample Data")
            st.dataframe(df.head(100), use_container_width=True, height=300)
    
    with tab2:
        st.markdown("### 🔗 URL Consolidation Analysis")
        
        if not st.session_state.analysis_complete:
            st.warning("⚠️ Please run analysis from the sidebar first!")
        else:
            url_consolidation = st.session_state.url_consolidation
            recommendations = url_consolidation['recommendations']
            summary = url_consolidation['summary']
            embeddings_used = url_consolidation['embeddings_used']
            
            # Show analysis type
            if embeddings_used:
                st.success("✅ Enhanced semantic analysis with embeddings")
            else:
                st.info("ℹ️ Basic semantic analysis (upload embeddings for enhanced results)")
            
            if len(recommendations) > 0:
                # Summary metrics
                st.markdown("### 📊 URL Consolidation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total URL Pairs", summary['total_pairs'])
                with col2:
                    st.metric("Total Potential Recovery", f"{summary['total_potential_recovery']:,}")
                with col3:
                    high_priority = summary['priorities'].get('High', 0)
                    st.metric("High Priority", high_priority)
                with col4:
                    medium_priority = summary['priorities'].get('Medium', 0)
                    st.metric("Medium Priority", medium_priority)
                
                # Action breakdown
                st.markdown("### 📋 Action Breakdown")
                
                actions = summary['actions']
                if actions:
                    cols = st.columns(min(len(actions), 4))
                    for idx, (action, count) in enumerate(actions.items()):
                        if idx < len(cols):
                            with cols[idx]:
                                st.metric(action, count)
                
                # Filter controls
                st.markdown("### 🔍 Filter URL Recommendations")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    action_filter = st.multiselect(
                        "Action",
                        options=['Merge', 'Redirect', 'Optimize', 'Internal Link', 'Monitor', 'Remove', 'False Positive'],
                        default=['Merge', 'Redirect', 'Optimize', 'Internal Link'],
                        key="action_filter"
                    )
                with col2:
                    min_recovery = st.number_input(
                        "Min Potential Recovery",
                        min_value=0,
                        value=10,
                        step=5,
                        key="min_recovery"
                    )
                with col3:
                    min_overlap = st.number_input(
                        "Min Keyword Overlap %",
                        min_value=0,
                        max_value=100,
                        value=5,
                        step=1,
                        key="min_overlap"
                    )
                
                # Filter recommendations
                filtered_recs = recommendations[
                    (recommendations['action'].isin(action_filter)) &
                    (recommendations['potential_recovery'] >= min_recovery) &
                    (recommendations['keyword_overlap_percentage'] >= min_overlap)
                ]
                
                if len(filtered_recs) > 0:
                    st.markdown("### 📊 Detailed URL Recommendations")
                    
                    # Download button
                    csv = filtered_recs.to_csv(index=False)
                    st.download_button(
                        label="📥 Download URL Recommendations",
                        data=csv,
                        file_name=f"url_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Display recommendations
                    display_cols = [
                        'primary_url', 'primary_indexed_queries', 'primary_clicks', 'primary_impressions',
                        'secondary_url', 'secondary_indexed_queries', 'secondary_clicks', 'secondary_impressions',
                        'semantic_similarity', 'keyword_overlap_count', 'keyword_overlap_percentage',
                        'recommended_action', 'priority'
                    ]
                    
                    st.dataframe(
                        filtered_recs[display_cols].sort_values('potential_recovery', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Individual recommendation details
                    st.markdown("### 🔍 Individual URL Recommendations")
                    for idx, rec in filtered_recs.head(10).iterrows():
                        with st.expander(f"🔗 {rec['action']}: {rec['primary_url']} ← {rec['secondary_url']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Primary Clicks", rec['primary_clicks'])
                                st.metric("Secondary Clicks", rec['secondary_clicks'])
                                st.metric("Potential Recovery", rec['potential_recovery'])
                            with col2:
                                st.metric("Keyword Overlap", f"{rec['keyword_overlap_count']} keywords")
                                st.metric("Overlap %", f"{rec['keyword_overlap_percentage']}%")
                                st.metric("Semantic Similarity", f"{rec['semantic_similarity']}%")
                            
                            st.info(f"**Action:** {rec['action']}")
                            st.info(f"**Priority:** {rec['priority']}")
                            
                            if len(rec['shared_keywords']) > 0:
                                st.write("**Shared Keywords:**")
                                st.write(", ".join(rec['shared_keywords'][:10]))
                                if len(rec['shared_keywords']) > 10:
                                    st.write(f"... and {len(rec['shared_keywords']) - 10} more")
                else:
                    st.info("No URL recommendations match the current filters. Try adjusting the filters.")
            else:
                st.info("No URL consolidation opportunities found.")

if __name__ == "__main__":
    main()
