"""
SEO Cannibalization Analysis Tool - Working Version
"""

import streamlit as st
import pandas as pd


def normalize_columns(df):
    """Normalize column names to standard format"""
    column_mapping = {
        'Query': 'query', 'Search Query': 'query',
        'Page': 'page', 'URL': 'page',
        'Clicks': 'clicks', 'Clicks (All)': 'clicks',
        'Impressions': 'impressions', 'Impressions (All)': 'impressions',
        'Position': 'position', 'Avg. Position': 'position',
    }
    
    new_columns = {}
    for col in df.columns:
        if col in column_mapping:
            new_columns[col] = column_mapping[col]
        else:
            col_lower = col.lower()
            for key, value in column_mapping.items():
                if key.lower() == col_lower:
                    new_columns[col] = value
                    break
            else:
                new_columns[col] = col
    
    return df.rename(columns=new_columns)


def clean_data(df):
    """Clean and prepare data"""
    required = ['query', 'page', 'clicks', 'impressions']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    
    df = df.copy()
    df['page'] = df['page'].astype(str)
    df['query'] = df['query'].astype(str)
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    
    df = df.dropna(subset=required)
    mask = df['page'].str.startswith('http')
    df = df[mask]
    
    return df


def analyze_cannibalization(gsc_df, similarity_df=None):
    """Cannibalization analysis - similarity file optional"""
    try:
        gsc_df = clean_data(gsc_df)
        if gsc_df is None:
            return None
        
        # Basic cannibalization analysis without similarity
        if similarity_df is None:
            # Find queries with multiple pages
            query_pages = gsc_df.groupby('query')['page'].nunique()
            multi_page_queries = query_pages[query_pages > 1].index
            
            if len(multi_page_queries) == 0:
                return pd.DataFrame()
            
            # Get data for multi-page queries
            cannibalization_df = gsc_df[gsc_df['query'].isin(multi_page_queries)]
            
            # Calculate metrics
            results = []
            for query in multi_page_queries:
                query_data = cannibalization_df[cannibalization_df['query'] == query]
                
                for _, row in query_data.iterrows():
                    results.append({
                        'query': query,
                        'page': row['page'],
                        'clicks': row['clicks'],
                        'impressions': row['impressions'],
                        'recommended_action': 'Review',
                        'priority': 'Medium'
                    })
            
            return pd.DataFrame(results)
        
        # With similarity file
        similarity_df = similarity_df.copy()
        
        # Auto-detect columns
        if len(similarity_df.columns) >= 3:
            similarity_df = similarity_df.iloc[:, :3]
            similarity_df.columns = ['url1', 'url2', 'similarity']
        else:
            st.warning("Similarity file needs 3 columns: url1, url2, similarity")
            return None
        
        similarity_df['url1'] = similarity_df['url1'].astype(str)
        similarity_df['url2'] = similarity_df['url2'].astype(str)
        similarity_df['similarity'] = pd.to_numeric(
            similarity_df['similarity'], errors='coerce'
        )
        
        gsc_urls = set(gsc_df['page'].unique())
        
        # Filter for URLs in GSC data
        mask = (
            similarity_df['url1'].isin(gsc_urls) &
            similarity_df['url2'].isin(gsc_urls)
        )
        similarity_df = similarity_df[mask].dropna()
        
        if similarity_df.empty:
            return pd.DataFrame()
        
        # Get metrics
        gsc_metrics = gsc_df.groupby('page').agg({
            'query': 'nunique',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        # Merge data
        merged = similarity_df.merge(
            gsc_metrics,
            left_on='url1',
            right_on='page',
            how='left'
        ).rename(columns={
            'query': 'url1_queries',
            'clicks': 'url1_clicks',
            'impressions': 'url1_impressions'
        })
        
        merged = merged.merge(
            gsc_metrics,
            left_on='url2',
            right_on='page',
            how='left',
            suffixes=('_1', '_2')
        ).rename(columns={
            'query_2': 'url2_queries',
            'clicks_2': 'url2_clicks',
            'impressions_2': 'url2_impressions'
        })
        
        # Determine primary URL (higher clicks)
        merged['primary_url'] = merged.apply(
            lambda x: x['url1'] if x['url1_clicks'] >= x['url2_clicks'] else x['url2'],
            axis=1
        )
        merged['secondary_url'] = merged.apply(
            lambda x: x['url2'] if x['url1_clicks'] >= x['url2_clicks'] else x['url1'],
            axis=1
        )
        
        # Reorder columns
        result = []
        for _, row in merged.iterrows():
            if row['url1_clicks'] >= row['url2_clicks']:
                primary_metrics = {
                    'queries': row['url1_queries'],
                    'clicks': row['url1_clicks'],
                    'impressions': row['url1_impressions']
                }
                secondary_metrics = {
                    'queries': row['url2_queries'],
                    'clicks': row['url2_clicks'],
                    'impressions': row['url2_impressions']
                }
            else:
                primary_metrics = {
                    'queries': row['url2_queries'],
                    'clicks': row['url2_clicks'],
                    'impressions': row['url2_impressions']
                }
                secondary_metrics = {
                    'queries': row['url1_queries'],
                    'clicks': row['url1_clicks'],
                    'impressions': row['url1_impressions']
                }
            
            result.append({
                'primary_url': row['primary_url'],
                'primary_queries': primary_metrics['queries'],
                'primary_clicks': primary_metrics['clicks'],
                'primary_impressions': primary_metrics['impressions'],
                'secondary_url': row['secondary_url'],
                'secondary_queries': secondary_metrics['queries'],
                'secondary_clicks': secondary_metrics['clicks'],
                'secondary_impressions': secondary_metrics['impressions'],
                'similarity_score': row['similarity'],
                'recommended_action': 'Merge' if row['similarity'] > 0.7 else 'Review',
                'priority': 'High' if row['similarity'] > 0.8 else 'Medium'
            })
        
        return pd.DataFrame(result)
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="SEO Cannibalization Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 SEO Cannibalization Analysis Tool")
    st.markdown("Analyze keyword cannibalization using GSC data")
    
    with st.sidebar:
        st.header("📁 Data Upload")
        
        gsc_file = st.file_uploader(
            "Upload GSC CSV/Excel file",
            type=['csv', 'xlsx'],
            help="Upload your Google Search Console export"
        )
        
        similarity_file = st.file_uploader(
            "Upload Similarity CSV/Excel file (optional)",
            type=['csv', 'xlsx'],
            help="Upload semantic similarity scores between URLs"
        )
        
        analyze_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)
    
    if gsc_file and analyze_button:
        with st.spinner("🔄 Processing data..."):
            try:
                if gsc_file.name.endswith('.csv'):
                    gsc_df = pd.read_csv(gsc_file)
                else:
                    gsc_df = pd.read_excel(gsc_file)
                
                similarity_df = None
                if similarity_file is not None:
                    if similarity_file.name.endswith('.csv'):
                        similarity_df = pd.read_csv(similarity_file)
                    else:
                        similarity_df = pd.read_excel(similarity_file)
                
                gsc_df = normalize_columns(gsc_df)
                results = analyze_cannibalization(gsc_df, similarity_df)
                
                if results is not None and not results.empty:
                    st.success("✅ Analysis completed successfully!")
                    
                    st.header("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Issues", len(results))
                    with col2:
                        st.metric("Unique Queries", results['query' if 'query' in results.columns else 'primary_url'].nunique())
                    with col3:
                        st.metric("Total Clicks", results['clicks' if 'clicks' in results.columns else 'primary_clicks'].sum())
                    
                    st.header("📋 Detailed Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="cannibalization_analysis.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.info("No cannibalization issues found. This could mean:")
                    st.markdown("- No queries have multiple competing pages")
                    st.markdown("- Data format needs adjustment")
                    st.markdown("- Try uploading just the GSC file first")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.info("Please check your file formats and try again.")
    
    else:
        st.info("👈 Please upload your GSC file to begin analysis. Similarity file is optional.")
        
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use
    1. **Export Google Search Console data** (Search Results > Export > CSV)
    2. **Upload your GSC CSV file** using the sidebar
    3. **Upload semantic similarity file** (optional, for enhanced analysis)
    4. **Click "Run Analysis"** to identify cannibalization issues
    
    ### 📊 Expected Output
    - **Queries** competing across multiple pages
    - **Traffic metrics** for each page
    - **Similarity scores** between competing pages (if similarity file provided)
    - **Actionable recommendations** for consolidation
    
    ### 📝 File Requirements
    - **GSC file**: Must have columns: query, page, clicks, impressions
    - **Similarity file**: Must have 3 columns: url1, url2, similarity_score
    """)


if __name__ == "__main__":
    main()
