"""
Data Input Page for SEO Cannibalization Analyzer
Handles both CSV upload and Google Search Console API integration
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from integrations.gsc_api import GSCConnector
from utils.state_manager import StateManager
from core.validators import DataValidator

def render(config: dict):
    """Render the data input page"""
    st.header("Data Source Selection")
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        render_gsc_section()
    
    with col2:
        render_csv_upload_section()
    
    # Show data preview if loaded
    if StateManager.is_data_loaded():
        render_data_preview()


def render_gsc_section():
    """Render Google Search Console connection section"""
    st.subheader("🔗 Google Search Console API")
    
    # Check if already authenticated
    if StateManager.get('gsc_authenticated'):
        st.success("✅ Connected to Google Search Console")
        
        # Property selection
        gsc_connector = GSCConnector()
        properties = gsc_connector.get_properties()
        
        if properties:
            selected_property = st.selectbox(
                "Select Property",
                options=properties,
                index=properties.index(StateManager.get('selected_property')) 
                if StateManager.get('selected_property') in properties else 0
            )
            StateManager.set('selected_property', selected_property)
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=90),
                max_value=datetime.now() - timedelta(days=1)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now() - timedelta(days=1)
            )
        
        # Dimension selection
        dimensions = st.multiselect(
            "Select Dimensions",
            options=['page', 'query', 'country', 'device'],
            default=['page', 'query']
        )
        
        # Filters
        with st.expander("Advanced Filters"):
            page_filter = st.text_input("Page URL contains:")
            query_filter = st.text_input("Query contains:")
            min_impressions = st.number_input("Minimum impressions:", min_value=0, value=10)
        
        # Fetch data button
        if st.button("Fetch Data", type="primary"):
            with st.spinner("Fetching data from Google Search Console..."):
                try:
                    # Build filters
                    filters = []
                    if page_filter:
                        filters.append({
                            'dimension': 'page',
                            'operator': 'contains',
                            'expression': page_filter
                        })
                    if query_filter:
                        filters.append({
                            'dimension': 'query',
                            'operator': 'contains',
                            'expression': query_filter
                        })
                    
                    # Fetch data
                    df = gsc_connector.fetch_data(
                        property_url=selected_property,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        dimensions=dimensions,
                        filters=filters
                    )
                    
                    # Filter by minimum impressions
                    df = df[df['impressions'] >= min_impressions]
                    
                    # Validate and store data
                    validator = DataValidator()
                    validation_result = validator.validate_dataframe(df)
                    
                    if validation_result['is_valid']:
                        StateManager.set_data(df)
                        st.success(f"✅ Loaded {len(df):,} rows of data")
                    else:
                        st.error(f"Data validation failed: {validation_result['errors']}")
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    else:
        # OAuth connection button
        if st.button("Connect to GSC", type="primary"):
            try:
                gsc_connector = GSCConnector()
                auth_url = gsc_connector.get_auth_url()
                
                st.markdown(f"[Click here to authorize]({auth_url})")
                
                # In a real implementation, handle OAuth callback
                auth_code = st.text_input("Enter authorization code:")
                
                if auth_code:
                    if gsc_connector.authenticate(auth_code):
                        StateManager.set('gsc_authenticated', True)
                        st.rerun()
                    else:
                        st.error("Authentication failed")
                        
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


def render_csv_upload_section():
    """Render CSV upload section"""
    st.subheader("📁 CSV Upload")
    
    uploaded_file = st.file_uploader(
        "Upload GSC Export",
        type=['csv'],
        help="CSV should contain: page, query, clicks, impressions, position"
    )
    
    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate data
            validator = DataValidator()
            validation_result = validator.validate_dataframe(df)
            
            if validation_result['is_valid']:
                StateManager.set_data(df)
                st.success(f"✅ Loaded {len(df):,} rows of data")
                
                # Show data quality metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Pages", f"{df['page'].nunique():,}")
                with col2:
                st.markdown("**Top 10 Pages by Clicks**")
                top_pages = df.groupby('page')['clicks'].sum().nlargest(10)
                # Truncate long URLs for display
                top_pages.index = [url[:50] + '...' if len(url) > 50 else url for url in top_pages.index]
                st.dataframe(
                    top_pages.reset_index(),
                    use_container_width=True,
                    hide_index=True
                ):
                    st.metric("Unique Queries", f"{df['query'].nunique():,}")
                with col3:
                    st.metric("Total Clicks", f"{df['clicks'].sum():,}")
            else:
                st.error("Data validation failed:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
                    
                # Show what's missing
                if validation_result['missing_columns']:
                    st.warning(f"Missing columns: {', '.join(validation_result['missing_columns'])}")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def render_data_preview():
    """Render data preview section"""
    df = StateManager.get_data()
    
    if df is not None:
        st.subheader("📊 Data Preview")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Quality", "Statistics"])
        
        with tab1:
            # Show sample data
            st.dataframe(
                df.head(100),
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            # Data quality checks
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Completeness**")
                missing_data = df.isnull().sum()
                if missing_data.sum() == 0:
                    st.success("✅ No missing data")
                else:
                    for col, count in missing_data[missing_data > 0].items():
                        st.warning(f"{col}: {count} missing values")
            
            with col2:
                st.markdown("**Data Types**")
                for col, dtype in df.dtypes.items():
                    st.text(f"{col}: {dtype}")
        
        with tab3:
            # Statistical summary
            st.markdown("**Statistical Summary**")
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(
                    df[numeric_cols].describe(),
                    use_container_width=True
                )
            
            # Top queries and pages
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 10 Queries by Clicks**")
                top_queries = df.groupby('query')['clicks'].sum().nlargest(10)
                st.dataframe(
                    top_queries.reset_index(),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2
