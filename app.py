import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="SEO Cannibalization Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 SEO Cannibalization Analysis Tool")
st.markdown("""
This tool analyzes keyword cannibalization issues using Google Search Console data. 
Upload your GSC data to identify when multiple pages compete for the same keywords.
""")

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("### Brand Keywords to Exclude")
brand_keywords = st.sidebar.text_area(
    "Enter brand keywords (one per line)",
    value="your-brand\nyourbrand\nyour brand",
    height=100,
    help="These keywords will be filtered out from the analysis"
)

# Convert brand keywords to list
if brand_keywords:
    brand_list = [keyword.strip().lower() for keyword in brand_keywords.split('\n') if keyword.strip()]
else:
    brand_list = []

# Thresholds
st.sidebar.markdown("### Analysis Thresholds")
click_threshold = st.sidebar.slider(
    "Click percentage threshold (%)",
    min_value=5,
    max_value=50,
    value=10,
    help="Minimum percentage of clicks for a page to be considered for cannibalization"
)

min_total_clicks = st.sidebar.number_input(
    "Minimum total clicks per query",
    min_value=1,
    value=10,
    help="Minimum total clicks a query must have to be analyzed"
)

# File upload
st.header("📂 Upload Your Google Search Console Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload your Google Search Console data with columns: page, query, clicks, impressions, position"
)

# Required columns
REQUIRED_COLUMNS = ['page', 'query', 'clicks', 'impressions', 'position']

def validate_data(df):
    """Validate the uploaded data"""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.info("Required columns: page, query, clicks, impressions, position")
        return False
    return True

def remove_branded_keywords(df, brand_list):
    """Remove queries containing branded keywords"""
    if not brand_list:
        return df
    
    # Create regex pattern for brand keywords
    brand_pattern = '|'.join([re.escape(brand) for brand in brand_list])
    
    # Filter out branded queries
    mask = ~df['query'].str.contains(brand_pattern, case=False, na=False)
    filtered_df = df[mask].copy()
    
    removed_count = len(df) - len(filtered_df)
    st.info(f"Removed {removed_count} rows containing branded keywords")
    
    return filtered_df

def analyze_cannibalization(df, click_threshold_pct, min_clicks):
    """Analyze keyword cannibalization"""
    
    # Convert to numeric
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    
    # Remove rows with missing data
    df = df.dropna(subset=['clicks', 'impressions', 'position'])
    
    # Calculate query-level metrics
    query_metrics = df.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'page': 'nunique'
    }).reset_index()
    
    query_metrics.columns = ['query', 'total_clicks', 'total_impressions', 'page_count']
    
    # Filter queries with multiple pages and minimum clicks
    multi_page_queries = query_metrics[
        (query_metrics['page_count'] > 1) & 
        (query_metrics['total_clicks'] >= min_clicks)
    ]
    
    if len(multi_page_queries) == 0:
        st.warning("No queries found with multiple pages and sufficient clicks.")
        return None, None
    
    # Merge back with original data
    df_filtered = df[df['query'].isin(multi_page_queries['query'])]
    df_analysis = df_filtered.merge(query_metrics, on='query', how='left')
    
    # Calculate percentages
    df_analysis['clicks_pct_vs_query'] = (df_analysis['clicks'] / df_analysis['total_clicks']) * 100
    df_analysis['page_clicks'] = df_analysis.groupby('page')['clicks'].transform('sum')
    df_analysis['clicks_pct_vs_page'] = (df_analysis['clicks'] / df_analysis['page_clicks']) * 100
    
    # Identify cannibalization opportunities
    threshold = click_threshold_pct / 100
    
    # Calculate how many pages per query have significant clicks
    df_analysis['significant_clicks'] = df_analysis['clicks_pct_vs_query'] >= click_threshold_pct
    significant_pages_per_query = df_analysis.groupby('query')['significant_clicks'].sum()
    
    # Mark potential opportunities
    df_analysis['cannibalization_risk'] = df_analysis['query'].map(
        lambda x: 'High' if significant_pages_per_query.get(x, 0) >= 2 else 'Low'
    )
    
    # Add recommendations
    def get_recommendation(row):
        if row['clicks_pct_vs_query'] >= click_threshold_pct and row['clicks_pct_vs_page'] >= click_threshold_pct:
            return "Potential Opportunity"
        else:
            return "Risk - Low Impact"
    
    df_analysis['recommendation'] = df_analysis.apply(get_recommendation, axis=1)
    
    return df_analysis, multi_page_queries

def create_summary_report(df_analysis):
    """Create summary statistics"""
    if df_analysis is None:
        return None
    
    summary = {
        'total_queries': df_analysis['query'].nunique(),
        'total_pages': df_analysis['page'].nunique(),
        'high_risk_queries': len(df_analysis[df_analysis['cannibalization_risk'] == 'High']['query'].unique()),
        'potential_opportunities': len(df_analysis[df_analysis['recommendation'] == 'Potential Opportunity']),
        'total_clicks_affected': df_analysis['total_clicks'].sum(),
        'avg_position': df_analysis['position'].mean()
    }
    
    return summary

# Main analysis logic
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        
        # Validate data
        if not validate_data(df):
            st.stop()
        
        # Show data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Remove branded keywords
        df_clean = remove_branded_keywords(df, brand_list)
        
        # Analyze cannibalization
        with st.spinner("Analyzing cannibalization..."):
            df_analysis, query_summary = analyze_cannibalization(
                df_clean, 
                click_threshold, 
                min_total_clicks
            )
        
        if df_analysis is not None:
            # Create summary report
            summary = create_summary_report(df_analysis)
            
            # Display summary
            st.header("📈 Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", summary['total_queries'])
            with col2:
                st.metric("High Risk Queries", summary['high_risk_queries'])
            with col3:
                st.metric("Potential Opportunities", summary['potential_opportunities'])
            with col4:
                st.metric("Total Affected Clicks", f"{summary['total_clicks_affected']:,}")
            
            # Display detailed results
            st.header("🔍 Detailed Analysis")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                risk_filter = st.selectbox(
                    "Filter by Risk Level",
                    ["All", "High", "Low"]
                )
            with col2:
                recommendation_filter = st.selectbox(
                    "Filter by Recommendation",
                    ["All", "Potential Opportunity", "Risk - Low Impact"]
                )
            
            # Apply filters
            display_df = df_analysis.copy()
            if risk_filter != "All":
                display_df = display_df[display_df['cannibalization_risk'] == risk_filter]
            if recommendation_filter != "All":
                display_df = display_df[display_df['recommendation'] == recommendation_filter]
            
            # Sort by clicks descending
            display_df = display_df.sort_values('clicks', ascending=False)
            
            # Display results
            st.dataframe(
                display_df[['query', 'page', 'clicks', 'impressions', 'position', 
                           'clicks_pct_vs_query', 'clicks_pct_vs_page', 'cannibalization_risk', 'recommendation']],
                use_container_width=True
            )
            
            # Top cannibalization issues
            st.header("🚨 Top Cannibalization Issues")
            high_risk_queries = df_analysis[df_analysis['cannibalization_risk'] == 'High']
            
            if len(high_risk_queries) > 0:
                top_issues = high_risk_queries.groupby('query').agg({
                    'clicks': 'sum',
                    'page': 'nunique',
                    'position': 'mean'
                }).reset_index().sort_values('clicks', ascending=False).head(10)
                
                st.dataframe(top_issues, use_container_width=True)
            else:
                st.info("No high-risk cannibalization issues found.")
            
            # Download options
            st.header("📥 Download Results")
            
            # Create Excel file with multiple sheets
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_analysis.to_excel(writer, sheet_name='Full Analysis', index=False)
                
                # Summary sheet
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # High risk queries
                if len(high_risk_queries) > 0:
                    high_risk_queries.to_excel(writer, sheet_name='High Risk', index=False)
                
                # Potential opportunities
                opportunities = df_analysis[df_analysis['recommendation'] == 'Potential Opportunity']
                if len(opportunities) > 0:
                    opportunities.to_excel(writer, sheet_name='Opportunities', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=output.getvalue(),
                file_name=f"seo_cannibalization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Recommendations
            st.header("💡 Recommendations")
            st.markdown("""
            **High Priority Actions:**
            1. Focus on queries marked as "High Risk" with multiple pages getting significant clicks
            2. Consider consolidating content for pages marked as "Potential Opportunity"
            3. Review internal linking strategy for cannibalized queries
            4. Implement 301 redirects where appropriate after content consolidation
            
            **Best Practices:**
            - Monitor cannibalization issues monthly
            - Update content strategy to avoid keyword overlap
            - Use canonical tags to indicate preferred pages
            - Optimize page titles and meta descriptions for clarity
            """)
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and columns.")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Show sample data format
    st.subheader("📋 Required Data Format")
    sample_data = pd.DataFrame({
        'page': ['https://example.com/page1', 'https://example.com/page2', 'https://example.com/page1'],
        'query': ['best hiking boots', 'best hiking boots', 'hiking boots review'],
        'clicks': [150, 45, 30],
        'impressions': [1000, 500, 300],
        'position': [3.2, 8.1, 12.5]
    })
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("""
    **How to export data from Google Search Console:**
    1. Go to Google Search Console
    2. Select your property
    3. Navigate to Performance > Search Results
    4. Set your date range (recommended: 3-6 months)
    5. Click on "Queries" and "Pages" tabs to enable both dimensions
    6. Click "Export" and select "Download CSV"
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | [GitHub Repository](https://github.com/SEOptimize-LLC/SEO-Cannibalization-Analysis)")
