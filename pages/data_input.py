"""
Data Input Page for SEO Cannibalization Analysis
Handles both file upload and Google Search Console direct connection
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
from column_mapper import normalize_column_names, validate_required_columns

# Google Search Console API settings
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
CLIENT_CONFIG = {
    "web": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [os.environ.get("REDIRECT_URI", "http://localhost:8501")]
    }
}

def init_oauth_flow():
    """Initialize OAuth flow for Google Search Console"""
    if not CLIENT_CONFIG["web"]["client_id"] or not CLIENT_CONFIG["web"]["client_secret"]:
        return None
    
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=CLIENT_CONFIG["web"]["redirect_uris"][0]
    )
    return flow

def get_gsc_data(credentials, site_url, start_date, end_date, row_limit=25000):
    """Fetch data from Google Search Console API"""
    service = build('searchconsole', 'v1', credentials=credentials)
    
    request = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'dimensions': ['page', 'query'],
        'rowLimit': row_limit,
        'dataState': 'FINAL'
    }
    
    try:
        response = service.searchanalytics().query(
            siteUrl=site_url, 
            body=request
        ).execute()
        
        if 'rows' in response:
            data = []
            for row in response['rows']:
                data.append({
                    'page': row['keys'][0],
                    'query': row['keys'][1],
                    'clicks': row['clicks'],
                    'impressions': row['impressions'],
                    'ctr': row['ctr'],
                    'position': row['position']
                })
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    except HttpError as error:
        st.error(f"An error occurred: {error}")
        return None

def get_site_list(credentials):
    """Get list of verified sites from Google Search Console"""
    service = build('searchconsole', 'v1', credentials=credentials)
    
    try:
        sites_list = service.sites().list().execute()
        return [site['siteUrl'] for site in sites_list.get('siteEntry', [])]
    except HttpError as error:
        st.error(f"An error occurred fetching sites: {error}")
        return []

def handle_oauth_callback():
    """Handle OAuth callback"""
    if 'code' in st.query_params:
        flow = init_oauth_flow()
        if flow:
            flow.fetch_token(code=st.query_params['code'])
            credentials = flow.credentials
            
            # Store credentials in session state
            st.session_state['gsc_credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Clear the code from URL
            st.query_params.clear()
            st.rerun()

def display_data_input():
    """Main function to display data input options"""
    st.title("📊 Data Input")
    st.markdown("Choose how to load your Google Search Console data")
    
    # Handle OAuth callback
    handle_oauth_callback()
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Upload CSV File", "Connect to Google Search Console"],
        key="data_source_radio"
    )
    
    if data_source == "Upload CSV File":
        display_file_upload()
    else:
        display_gsc_connection()

def display_file_upload():
    """Display file upload interface"""
    st.markdown("### 📁 Upload CSV File")
    st.markdown("""
    Upload a CSV export from Google Search Console with the following columns:
    - **page** (or url, landing_page)
    - **query** (or keyword, search_query)
    - **clicks**
    - **impressions**
    - **position** (or average_position, rank)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Select your Google Search Console export file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Normalize column names
            df = normalize_column_names(df)
            
            # Validate columns
            is_valid, missing_columns = validate_required_columns(df)
            
            if not is_valid:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Your file has these columns: " + ", ".join(df.columns))
                return
            
            # Store in session state
            st.session_state['gsc_data'] = df
            st.session_state['data_loaded'] = True
            st.session_state['data_source'] = 'file'
            
            # Display success message and data preview
            st.success(f"✅ Data loaded successfully! {len(df):,} rows found.")
            
            # Display data preview
            with st.expander("📋 Data Preview", expanded=True):
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Unique Pages", f"{df['page'].nunique():,}")
                with col3:
                    st.metric("Unique Queries", f"{df['query'].nunique():,}")
                with col4:
                    st.metric("Total Clicks", f"{df['clicks'].sum():,}")
                
                # Top pages by clicks
                st.markdown("**Top 10 Pages by Clicks**")
                top_pages = df.groupby('page')['clicks'].sum().sort_values(ascending=False).head(10)
                st.dataframe(
                    top_pages.reset_index().rename(columns={'page': 'Page', 'clicks': 'Total Clicks'}),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Sample data
                st.markdown("**Sample Data (first 100 rows)**")
                st.dataframe(df.head(100), use_container_width=True, height=300)
            
            # Navigation button
            if st.button("🚀 Proceed to Analysis", type="primary", use_container_width=True):
                st.switch_page("pages/analysis.py")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def display_gsc_connection():
    """Display Google Search Console connection interface"""
    st.markdown("### 🔗 Connect to Google Search Console")
    
    # Check if OAuth is configured
    flow = init_oauth_flow()
    if not flow:
        st.info("""
        📌 **Direct GSC connection requires API setup.**
        
        To enable this feature:
        1. Set up a Google Cloud Project
        2. Enable the Search Console API
        3. Configure OAuth 2.0 credentials
        
        **For now, please use the CSV upload option above, which provides the same functionality.**
        """)
        
        with st.expander("📖 Detailed Setup Instructions"):
            st.markdown("""
            ### Setting up Google Search Console API
            
            1. **Create a Google Cloud Project:**
               - Visit [Google Cloud Console](https://console.cloud.google.com/)
               - Create a new project
               - Enable "Google Search Console API"
            
            2. **Create OAuth Credentials:**
               - Go to APIs & Services > Credentials
               - Create OAuth 2.0 Client ID (Web application)
               - Add redirect URI: `https://your-app.streamlit.app`
            
            3. **Set Environment Variables:**
               ```
               GOOGLE_CLIENT_ID = "your-client-id"
               GOOGLE_CLIENT_SECRET = "your-secret"
               REDIRECT_URI = "your-redirect-uri"
               ```
            """)
        return
    
    # Check if already authenticated
    if 'gsc_credentials' in st.session_state:
        display_authenticated_interface()
    else:
        # Display authentication button
        st.markdown("""
        Connect directly to Google Search Console to fetch your data automatically.
        This requires authentication with your Google account.
        """)
        
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account'
        )
        
        st.markdown("""
        To connect to Google Search Console, click the button below. 
        
        **Note:** This will open in a new tab to comply with Google's security requirements.
        """)
        
        # Use a link that opens in a new tab instead of iframe
        st.markdown(f'''
        <a href="{auth_url}" target="_blank">
            <button style="
                background-color: #4285F4;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            ">
                🔐 Connect to Google Search Console
            </button>
        </a>
        ''', unsafe_allow_html=True)
        
        # Add instructions for users
        st.info("""
        **After authentication:**
        1. Complete the Google sign-in process in the new tab
        2. You'll be redirected back to this app
        3. If the redirect doesn't work, copy the URL from the browser and paste it here:
        """)
        
        # Manual URL input as fallback
        auth_response_url = st.text_input("Paste the redirect URL here (if needed):", key="auth_url_input")
        
        if auth_response_url and 'code=' in auth_response_url:
            try:
                # Extract the code from the URL
                import urllib.parse
                parsed_url = urllib.parse.urlparse(auth_response_url)
                code = urllib.parse.parse_qs(parsed_url.query).get('code', [None])[0]
                
                if code:
                    # Process the code
                    flow.fetch_token(code=code)
                    credentials = flow.credentials
                    
                    # Store credentials in session state
                    st.session_state['gsc_credentials'] = {
                        'token': credentials.token,
                        'refresh_token': credentials.refresh_token,
                        'token_uri': credentials.token_uri,
                        'client_id': credentials.client_id,
                        'client_secret': credentials.client_secret,
                        'scopes': credentials.scopes
                    }
                    
                    st.success("✅ Successfully connected to Google Search Console!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing authentication: {str(e)}")

def display_authenticated_interface():
    """Display interface for authenticated users"""
    # Reconstruct credentials
    creds_data = st.session_state['gsc_credentials']
    credentials = Credentials(
        token=creds_data['token'],
        refresh_token=creds_data['refresh_token'],
        token_uri=creds_data['token_uri'],
        client_id=creds_data['client_id'],
        client_secret=creds_data['client_secret'],
        scopes=creds_data['scopes']
    )
    
    st.success("✅ Connected to Google Search Console")
    
    # Logout button
    if st.button("🔓 Disconnect", type="secondary"):
        del st.session_state['gsc_credentials']
        st.rerun()
    
    # Get site list
    sites = get_site_list(credentials)
    
    if not sites:
        st.error("No verified sites found in your Google Search Console account.")
        return
    
    # Site selection
    selected_site = st.selectbox(
        "Select a property:",
        options=sites,
        help="Choose the website you want to analyze"
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90),
            max_value=datetime.now() - timedelta(days=1),
            help="Select the start date for data collection"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now() - timedelta(days=1),
            help="Select the end date for data collection"
        )
    
    # Row limit
    row_limit = st.number_input(
        "Row Limit",
        min_value=1000,
        max_value=25000,
        value=25000,
        step=1000,
        help="Maximum number of rows to fetch (API limit: 25,000)"
    )
    
    # Fetch data button
    if st.button("📥 Fetch Data", type="primary", use_container_width=True):
        with st.spinner("Fetching data from Google Search Console..."):
            df = get_gsc_data(credentials, selected_site, start_date, end_date, row_limit)
            
            if df is not None and not df.empty:
                # Store in session state
                st.session_state['gsc_data'] = df
                st.session_state['data_loaded'] = True
                st.session_state['data_source'] = 'api'
                st.session_state['selected_site'] = selected_site
                st.session_state['date_range'] = (start_date, end_date)
                
                st.success(f"✅ Data fetched successfully! {len(df):,} rows retrieved.")
                
                # Display data preview
                with st.expander("📋 Data Preview", expanded=True):
                    # Basic statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Unique Pages", f"{df['page'].nunique():,}")
                    with col3:
                        st.metric("Unique Queries", f"{df['query'].nunique():,}")
                    with col4:
                        st.metric("Total Clicks", f"{df['clicks'].sum():,}")
                    
                    # Top pages by clicks
                    st.markdown("**Top 10 Pages by Clicks**")
                    top_pages = df.groupby('page')['clicks'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(
                        top_pages.reset_index().rename(columns={'page': 'Page', 'clicks': 'Total Clicks'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Sample data
                    st.markdown("**Sample Data (first 100 rows)**")
                    st.dataframe(df.head(100), use_container_width=True, height=300)
                
                # Navigation button
                if st.button("🚀 Proceed to Analysis", type="primary", use_container_width=True):
                    st.switch_page("pages/analysis.py")
            else:
                st.error("No data retrieved. Please check your date range and try again.")

# Main execution
if __name__ == "__main__":
    display_data_input()
