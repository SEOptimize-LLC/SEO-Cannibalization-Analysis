"""
Simple Streamlit App for SEO Cannibalization Analysis
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


def analyze_cannibalization(gsc_df, similarity_df):
    """Simple cannibalization analysis"""
    try:
        gsc_df = clean_data(gsc_df)
        if gsc_df is None:
            return None
        
        similarity_df = similarity_df.copy()
        
        # Use first 3 columns regardless of column names
        if len(similarity_df.columns) >= 3:
            similarity_df = similarity_df.iloc[:, :3]
            similarity_df.columns = [
                'primary_url', 'secondary_url', 'similarity_score'
            ]
        else:
            st.error("Similarity file needs at least 3 columns")
            return None
        
        similarity_df['primary_url'] = similarity_df['primary_url'].astype(str)
        similarity_df['secondary_url'] = similarity_df['secondary_url'].astype(str)
        similarity_df['similarity_score'] = pd.to_numeric(
            similarity_df['similarity_score'], errors='coerce'
        )
        
        gsc_urls = set(gsc_df['page'].unique())
        
        mask = (
            similarity_df['primary_url'].isin(gsc_urls) &
            similarity_df['secondary_url'].isin(gsc_urls)
        )
        similarity_df = similarity_df[mask].dropna()
        
        gsc_metrics = gsc_df.groupby('page').agg({
            'query': 'nunique',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        merged = similarity_df.merge(
            gsc_metrics,
            left_on='primary_url',
            right_on='page',
            how='left'
        ).rename(columns={
            'query': 'primary_queries',
            'clicks': 'primary_clicks',
            'impressions': 'primary_impressions'
        })
        
        merged = merged.merge(
            gsc_metrics,
            left_on='secondary_url',
            right_on='page',
            how='left',
            suffixes=('_primary', '_secondary')
        ).rename(columns={
            'query_secondary': 'secondary_queries',
            'clicks_secondary': 'secondary_clicks',
            'impressions_secondary': 'secondary_impressions'
        })
        
        merged['recommended_action'] = 'Analyze'
        merged['priority'] = 'Medium'
        
        result = merged[[
            'primary_url', 'secondary_url', 'similarity_score',
            'primary_queries', 'primary_clicks', 'primary_impressions',
            'secondary_queries', 'secondary_clicks', 'secondary_impressions',
            'recommended_action', 'priority'
        ]].dropna()
        
        return result
        
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
            "Upload Similarity CSV/Excel file",
            type=['csv', 'xlsx'],
            help="Upload semantic similarity scores between URLs"
        )
        
        analyze_button = st.button("🔬 Run Analysis", type="primary", use_container_width=True)
    
    if gsc_file and similarity_file and analyze_button:
        with st.spinner("🔄 Processing data..."):
            try:
                if gsc_file.name.endswith('.csv'):
                    gsc_df = pd.read_csv(gsc_file)
                else:
                    gsc_df = pd.read_excel(gsc_file)
                
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
                        st.metric("Total URL Pairs", len(results))
                    with col2:
                        st.metric("Unique Primary URLs", results['primary_url'].nunique())
                    with col3:
                        st.metric("Unique Secondary URLs", results['secondary_url'].nunique())
                    
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
                    st.warning("No cannibalization issues found or data format error.")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    
    else:
        st.info("👈 Please upload both GSC and similarity files using the sidebar to begin analysis.")
        
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use
    1. **Export Google Search Console data** (Search Results > Export > CSV)
    2. **Upload your GSC CSV file** using the sidebar
    3. **Upload semantic similarity file** (optional but recommended)
    4. **Click "Run Analysis"** to identify cannibalization issues
    
    ### 📊 Expected Output
    - **URL pairs** with potential cannibalization
    - **Traffic metrics** for each URL
    - **Similarity scores** between competing pages
    - **Actionable recommendations** for consolidation
    """)


if __name__ == "__main__":
    main()
