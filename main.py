"""
main.py — SEO Cannibalization Analyzer
v2.1.1 • 2025-07-23

Changelog
---------
✓ Always write analysis_results, even when empty  
✓ User-friendly “no data” banners in Analysis, Visualizations, Recommendations and Export tabs  
✓ Added hint links so users can relax thresholds without re-uploading the CSV
"""

from __future__ import annotations
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import helpers
from column_mapper import validate_and_clean

# ─────────────────────────────────────────────────────────────
# Initial page & session config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEO Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "data_loaded": False,
    "analysis_complete": False,
    "df": None,
    "analysis_results": {},
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("🔍 SEO Cannibalization Analyzer")
st.markdown("### Advanced Detection & Resolution Tool")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Sidebar – config panel
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Detection Thresholds")
    click_threshold = st.slider(
        "Minimum Click % Threshold", 1, 20, 5, help="Page share of query clicks"
    ) / 100
    min_clicks = st.number_input(
        "Minimum Total Clicks", 1, value=10, help="Filter out very low-click queries"
    )

    st.subheader("Brand Exclusions")
    brand_terms = st.text_area("Brand terms (one per line)")
    brand_list = [t.strip() for t in brand_terms.splitlines() if t.strip()]

    st.subheader("Advanced Analysis")
    enable_intent_detection = st.checkbox("Intent mismatch", True)
    enable_serp_analysis = st.checkbox("SERP feature analysis", True)
    enable_content_gap = st.checkbox("Content-gap analysis", True)
    enable_ml_insights = st.checkbox("AI-powered insights", True)

# ─────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data Input", "🔍 Analysis", "📈 Visualizations", "🎯 Recommendations", "📥 Export"]
)

# ─────────────────────────────────────────────────────────────
# Tab 1 – CSV upload
# ─────────────────────────────────────────────────────────────
with tab1:
    st.header("Data Source")
    upf = st.file_uploader(
        "Upload Google Search Console CSV",
        type=["csv"],
        help="Headers like ‘Landing Page’ or ‘Avg. Pos’ are mapped automatically.",
    )

    if upf:
        try:
            raw_df = pd.read_csv(upf, low_memory=False)
            df, mapping, missing = validate_and_clean(raw_df)

            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()

            st.session_state.df = df
            st.session_state.data_loaded = True

            st.success(
                f"✅ Parsed **{len(df):,}** rows – all required fields recognised."
            )
            with st.expander("Preview first 5 rows", expanded=False):
                st.dataframe(df.head())

            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Pages", f"{df['page'].nunique():,}")
            c2.metric("Unique Queries", f"{df['query'].nunique():,}")
            c3.metric("Total Clicks", f"{int(df['clicks'].sum()):,}")

        except Exception as exc:
            st.error(f"Error loading file: {exc}")

# ─────────────────────────────────────────────────────────────
# Tab 2 – analysis pipeline
# ─────────────────────────────────────────────────────────────
with tab2:
    st.header("Cannibalization Analysis")

    if st.session_state.data_loaded:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Crunching numbers…"):
                df = st.session_state.df
                progress = st.progress(0)

                # 1 Remove brand queries
                progress.progress(15)
                df_nb = helpers.remove_brand_queries(df, brand_list) if brand_list else df

                # 2 Metrics per (query, page)
                progress.progress(30)
                qp = helpers.calculate_query_page_metrics(df_nb)

                # 3 Filter by click/page thresholds
                progress.progress(50)
                qc = helpers.filter_queries_by_clicks_and_pages(
                    qp, min_clicks=min_clicks
                )

                # 4 Aggregate & compute percentages
                progress.progress(65)
                wip = helpers.merge_and_aggregate(qp, qc)
                wip = helpers.calculate_click_percentage(wip)
                wip["clicks_pct_vs_query"] = wip.groupby("query")["clicks"].transform(
                    lambda x: x / x.sum() if x.sum() else 0
                )

                # 5 Threshold filter
                progress.progress(80)
                keep = (
                    wip[wip["clicks_pct_vs_query"] >= click_threshold]
                    .groupby("query")
                    .filter(lambda x: len(x) >= 2)["query"]
                    .unique()
                )
                wip = wip[wip["query"].isin(keep)]

                # 6 Merge page-level clicks
                progress.progress(90)
                wip = helpers.merge_with_page_clicks(wip, df)

                # 7 Opportunity scoring & sort
                progress.progress(98)
                final_df = helpers.sort_and_finalize_output(
                    helpers.define_opportunity_levels(wip)
                )

                # Ensure results object exists even if empty
                st.session_state.analysis_results = {
                    "all_opportunities": final_df,
                    "immediate_opportunities": helpers.immediate_opps(final_df),
                    "qa_data": helpers.create_qa_dataframe(df, final_df),
                }
                st.session_state.analysis_complete = True
                progress.progress(100)

                st.success("✅ Analysis complete!")

    # Show results if available
    if st.session_state.analysis_complete:
        results_df = st.session_state.analysis_results["all_opportunities"]
        if results_df.empty:
            st.warning(
                "No cannibalization issues detected with the current settings. "
                "Try lowering the *Minimum Click % Threshold* or *Minimum Total Clicks* "
                "in the sidebar and re-run the analysis."
            )
        else:
            st.subheader("Cannibalization Issues Detected")
            selected_queries = st.multiselect(
                "Filter by Query",
                options=results_df["query"].unique(),
                key="query_filter",
            )
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "Potential Opportunity", "Risk"],
                key="status_filter",
            )

            filtered = results_df.copy()
            if selected_queries:
                filtered = filtered[filtered["query"].isin(selected_queries)]
            if status_filter != "All":
                filtered = filtered[filtered["comment"].str.contains(status_filter)]

            st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.info("Upload data and run the analysis first.")

# ─────────────────────────────────────────────────────────────
# Tab 3 – visualizations
# ─────────────────────────────────────────────────────────────
with tab3:
    st.header("Visual Analysis")
    if st.session_state.analysis_complete:
        data = st.session_state.analysis_results["all_opportunities"]
        if data.empty:
            st.warning("No visualization available – the analysis returned no rows.")
            st.stop()

        c1, c2 = st.columns(2)

        # Distribution bar
        pages_per_q = data.groupby("query")["page"].count().value_counts().sort_index()
        fig_dist = go.Figure(
            go.Bar(
                x=[f"{n} Pages" for n in pages_per_q.index],
                y=pages_per_q.values,
                marker_color=[
                    "#28a745" if n == 2 else "#ffc107" if n == 3 else "#dc3545"
                    for n in pages_per_q.index
                ],
            )
        )
        fig_dist.update_layout(
            title="Cannibalization Distribution",
            xaxis_title="Competing Pages per Query",
            yaxis_title="Number of Queries",
            height=400,
        )
        c1.plotly_chart(fig_dist, use_container_width=True)

        # Opportunity vs risk pie
        com = data["comment"].value_counts()
        fig_pie = go.Figure(
            go.Pie(
                labels=com.index,
                values=com.values,
                hole=0.3,
                marker_colors=[
                    "#28a745" if "Opportunity" in x else "#dc3545" for x in com.index
                ],
            )
        )
        fig_pie.update_layout(title="Opportunity vs Risk", height=400)
        c2.plotly_chart(fig_pie, use_container_width=True)

        # Click distribution
        st.subheader("Top 10 Cannibalised Queries")
        top_q = data.groupby("query")["clicks_query"].sum().nlargest(10).index
        fig_clicks = px.bar(
            data[data["query"].isin(top_q)],
            x="query",
            y="clicks_query",
            color="page",
            labels={"clicks_query": "Clicks"},
            title="Click Distribution for Top Queries",
        )
        fig_clicks.update_layout(height=500)
        st.plotly_chart(fig_clicks, use_container_width=True)
    else:
        st.info("Run the analysis first to generate visualisations.")

# ─────────────────────────────────────────────────────────────
# Tab 4 – recommendations
# ─────────────────────────────────────────────────────────────
with tab4:
    st.header("Consolidation Recommendations")
    if st.session_state.analysis_complete:
        imm = st.session_state.analysis_results["immediate_opportunities"]
        if imm.empty:
            st.info(
                "No immediate consolidation opportunities identified with current "
                "thresholds. Adjust thresholds if you expected results."
            )
        else:
            st.subheader("🎯 High-Priority Opportunities")
            for q in imm["query"].unique():
                with st.expander(f"Query: {q}"):
                    qd = imm[imm["query"] == q].sort_values(
                        "clicks_query", ascending=False
                    )
                    primary = qd.iloc[0]
                    st.info(f"🏆 {primary['page']}")
                    st.write(f"Clicks → {primary['clicks_query']:,}")
                    st.write(f"Avg Position → {primary['avg_position']:.1f}")

                    st.markdown("**Pages to Redirect (301):**")
                    for _, row in qd.iloc[1:].iterrows():
                        st.write(f"↪️ {row['page']} – {row['clicks_query']:,} clicks")

                    total = qd["clicks_query"].sum()
                    recover = int((total - primary["clicks_query"]) * 0.7)
                    st.success(f"Potential recovery ≈ +{recover:,} clicks/month")
    else:
        st.info("Analyse data to see recommendations.")

# ─────────────────────────────────────────────────────────────
# Tab 5 – export
# ─────────────────────────────────────────────────────────────
with tab5:
    st.header("Export Results")
    if st.session_state.analysis_complete:
        res = st.session_state.analysis_results["all_opportunities"]
        if res.empty:
            st.info("Nothing to export – the analysis returned no rows.")
            st.stop()

        c1, c2 = st.columns(2)

        # Excel workbook
        with BytesIO() as buf:
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                st.session_state.analysis_results["all_opportunities"].to_excel(
                    xw, sheet_name="all_potential_opps", index=False
                )
                st.session_state.analysis_results["immediate_opportunities"].to_excel(
                    xw, sheet_name="high_likelihood_opps", index=False
                )
                st.session_state.analysis_results["qa_data"].to_excel(
                    xw, sheet_name="risk_qa_data", index=False
                )
            buf.seek(0)
            c1.download_button(
                "📊 Download Full Report (Excel)",
                buf,
                file_name=f"seo_cannibalization_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

        # Redirect CSV
        imm = st.session_state.analysis_results["immediate_opportunities"]
        if not imm.empty:
            redir_rows = []
            for q in imm["query"].unique():
                qd = imm[imm["query"] == q].sort_values(
                    "clicks_query", ascending=False
                )
                primary = qd.iloc[0]["page"]
                for _, row in qd.iloc[1:].iterrows():
                    redir_rows.append(
                        {
                            "url_from": row["page"],
                            "url_to": primary,
                            "query": q,
                            "clicks_to_recover": row["clicks_query"],
                        }
                    )
            csv = pd.DataFrame(redir_rows).to_csv(index=False)
            c2.download_button(
                "🔄 Download Redirect Map (CSV)",
                csv,
                file_name=f"redirect_map_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )
        else:
            c2.info("No redirects to export.")
    else:
        st.info("Run the analysis first.")

# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666">'
    "SEO Cannibalization Analyzer v2.1.1 | Built with ❤️ for SEO professionals"
    "</div>",
    unsafe_allow_html=True,
)
