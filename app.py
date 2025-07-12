"""
SEO Cannibalization Analyzer - Main Application Entry Point
"""

import streamlit as st
from config.settings import APP_CONFIG
from utils.state_manager import initialize_session_state
from ui.layouts import render_sidebar
from ui.styles import load_custom_css
from pages import data_input, analysis, visualizations, recommendations, export

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG['app_name'],
    page_icon=APP_CONFIG['app_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state=APP_CONFIG['sidebar_state']
)

# Initialize session state
initialize_session_state()

# Load custom CSS
load_custom_css()

# Main app header
st.title(f"{APP_CONFIG['app_icon']} {APP_CONFIG['app_name']}")
st.markdown(f"### {APP_CONFIG['app_description']}")

# Render sidebar configuration
config = render_sidebar()

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Input", 
    "🔍 Analysis", 
    "📈 Visualizations", 
    "🎯 Recommendations", 
    "📥 Export"
])

# Render each page
with tab1:
    data_input.render(config)

with tab2:
    analysis.render(config)

with tab3:
    visualizations.render(config)

with tab4:
    recommendations.render(config)

with tab5:
    export.render(config)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        {APP_CONFIG['app_name']} v{APP_CONFIG['version']} | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
