import streamlit as st

# --- Main application entry point ---
def main():
    st.set_page_config(page_title="SEO Cannibalization Analysis", layout="wide")
    st.title("SEO Cannibalization Analysis")

    # Authentication step
    if "credentials" not in st.session_state:
        if st.button("Authenticate with Google Search Console"):
            st.session_state.credentials = authenticate()
    else:
        run_analysis_ui()

# --- UI for analysis parameters and results ---
def run_analysis_ui():
    from datetime import date

    st.sidebar.header("Data & Settings")
    start_date = st.sidebar.date_input("Start Date", date.today().replace(day=1))
    end_date = st.sidebar.date_input("End Date", date.today())
    threshold = st.sidebar.slider("Cannibalization Threshold", 0.0, 1.0, 0.5)

    if st.sidebar.button("Load & Analyze"):
        df = load_gsc_data(st.session_state.credentials, start_date, end_date)
        results = process_cannibalization(df, threshold)
        st.subheader("Analysis Results")
        st.dataframe(results, use_container_width=True)
        fig = plot_results(results)
        st.plotly_chart(fig, use_container_width=True)

# --- Authentication function (lazy import) ---
def authenticate():
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build

    flow = Flow.from_client_secrets_file(
        "config/client_secrets.json",
        scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
        redirect_uri="urn:ietf:wg:oauth:2.0:oob"
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    st.info(f"Go to this URL to authorize: {auth_url}")
    code = st.text_input("Enter authorization code")
    if code:
        flow.fetch_token(code=code)
        creds = flow.credentials
        return creds
    st.stop()

# --- Cached data loading ---
@st.cache_data(show_spinner=False)
def load_gsc_data(credentials, start_date, end_date):
    import pandas as pd
    from googleapiclient.discovery import build

    service = build("webmasters", "v3", credentials=credentials)
    request = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": ["page", "query"],
        "rowLimit": 10000
    }
    response = service.searchanalytics().query(
        siteUrl="https://example.com", body=request
    ).execute()
    rows = response.get("rows", [])
    data = []
    for r in rows:
        data.append({
            "page": r["keys"][0],
            "query": r["keys"][1],
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0)
        })
    df = pd.DataFrame(data)
    return df

# --- Cached processing ---
@st.cache_data
def process_cannibalization(df, threshold):
    import pandas as pd
    import numpy as np

    pivot = df.pivot_table(index="page", values="impressions", aggfunc="sum").reset_index()
    pivot["score"] = pivot["impressions"] / pivot["impressions"].max()
    pivot["cannibalization"] = pivot["score"] >= threshold
    return pivot.sort_values("score", ascending=False)

# --- Cached plotting ---
@st.cache_data
def plot_results(df):
    import plotly.express as px

    fig = px.bar(
        df,
        x="page",
        y="score",
        color="cannibalization",
        labels={
            "score": "Cannibalization Score",
            "cannibalization": "Above Threshold"
        },
        title="Page-Level Cannibalization Scores"
    )
    return fig

if __name__ == "__main__":
    main()
