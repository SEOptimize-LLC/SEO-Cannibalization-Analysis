"""
Simple Streamlit App for SEO Cannibalization Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import re


def normalize_columns(df):
    """Normalize column names to standard format"""
    column_mapping = {
        'Query': 'query', 'Search Query': 'query', 'search_query': 'query',
        'Page': 'page', 'URL': 'page', 'page_url': 'page',
        'Clicks': 'clicks', 'Clicks (All)': 'clicks', 'clicks_all': 'clicks',
        'Impressions': 'impressions', 'Impressions (All)': 'impressions',
        'impressions_all': 'impressions',
        'Position': 'position', 'Avg. Position': 'position', 'avg_position': 'position',
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
    # Ensure required columns exist
    required = ['query', 'page', 'clicks', 'impressions']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    
    # Clean data
    df = df.copy()
    df['page'] = df['page'].astype(str)
    df['query'] = df['query'].astype(str)
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    
    # Remove NaN values
    df = df.dropna(subset=required)
    
    # Filter out invalid URLs
    mask = df['page'].str.startswith('http')
    df = df[mask]
    
    return df


def analyze_cannibalization(gsc_df, similarity_df):
    """Simple cannibalization analysis"""
    try:
        # Clean GSC data
        gsc_df = clean_data(gsc_df)
        if gsc_df is None:
            return None
        
        # Clean similarity data
        similarity_df = similarity_df.copy()
        
        # Ensure similarity columns exist
        if len(similarity_df.columns) >= 3:
            similarity_df.columns = ['primary_url', 'secondary_url', 'similarity_score']
        else:
            st.error("Similarity file needs at least 3 columns")
            return None
        
        # Clean similarity data
        similarity_df['primary_url'] = similarity_df['primary_url'].astype(str)
        similarity_df['secondary_url'] = similarity_df['secondary_url'].astype(str)
        similarity_df['similarity_score'] = pd.to_numeric(
            similarity_df['similarity_score'], errors='coerce'
        )
        
        # Get GSC URLs
        gsc_urls = set(gsc_df['page'].unique())
        
        # Filter valid URL pairs
        mask = (
            similarity_df['primary_url'].isin(gsc_urls) & 
            similarity_df['secondary_url'].isin(gsc_urls)
        )
        similarity_df = similarity_df[mask].dropna()
        
        # Aggregate GSC metrics
        gsc_metrics = gsc_df.groupby('page').agg({
            'query': 'nunique',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        # Merge data
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
        
        # Add recommendations
        merged['recommended_action'] = 'Analyze'
        merged['priority'] = 'Medium'
        
        # Clean up
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
    st.markdown("Analyze keyword cannibalization using GSC data and semantic similarity")
    
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
                
                # Normalize GSC columns
                gsc_df = normalize_columns(gsc_df)
                
                # Run analysis
                results = analyze_cannibalization(gsc_df, similarity_df)
                
                if results is not None and not results.empty:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Summary statistics
                    st.header("📊 Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total URL Pairs", len(results))
                    with col2:
                        st.metric("Unique Primary URLs", results['primary_url'].nunique())
                    with col3:
                        st.metric("Unique Secondary URLs", results['secondary_url'].nunique())
                    
                    # Detailed results
                    st.header("🔍 Detailed Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
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
