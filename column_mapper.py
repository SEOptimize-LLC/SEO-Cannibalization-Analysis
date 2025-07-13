"""
column_mapper.py
----------------
Utility functions to normalise Google Search Console exports for the
SEO Cannibalization Analyzer.

Key Features
============
1. Robust header mapping:
   - Handles case, pluralisation, common synonyms (“Landing Page” → page,
     “Avg. Pos” → position, etc.).
2. Numeric coercion:
   - Guarantees clicks, impressions, and position arrive as numeric dtypes.
3. Minimal public surface:
   - `validate_and_clean(df)` is the canonical entry-point used by main.py.

© 2025 SEOptimize LLC – MIT License
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
_HEADER_ALIASES: Dict[str, List[str]] = {
    "page": [
        "page", "pages", "url", "urls", "landing page", "landing pages",
        "landingpage", "landingpages", "page_url", "page url", "pageurl",
        "destination", "destinations", "link", "links", "webpage", "webpages",
        "site", "sites", "page_path", "pagepath", "path",
    ],
    "query": [
        "query", "queries", "keyword", "keywords", "search term", "search terms",
        "searchterm", "searchterms", "search query", "search queries",
        "searchquery", "searchqueries", "term", "terms", "phrase", "phrases",
        "top_queries", "top queries",
    ],
    "clicks": [
        "clicks", "click", "total clicks", "totalclicks", "click count",
        "clickcount", "click_count", "ctr clicks", "ctrclicks", "total_clicks",
    ],
    "impressions": [
        "impressions", "impression", "total impressions", "totalimpressions",
        "impression count", "impressioncount", "impression_count",
        "views", "view", "total views", "totalviews", "total_impressions",
    ],
    "position": [
        "position", "positions", "avg position", "avgposition", "avg_position",
        "average position", "averageposition", "average_position",
        "avg. position", "avg. pos", "avg pos", "avgpos", "avg.pos",
        "rank", "ranking", "rankings", "avg rank", "avgrank", "avg_rank",
        "average rank", "averagerank", "average_rank", "avg_ranking",
    ],
}


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation/whitespace, collapse runs of spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\.]", " ", text.lower())).strip()


def _build_lookup() -> Dict[str, str]:
    """Create reverse lookup table alias → canonical header."""
    alias_map: Dict[str, str] = {}
    for canonical, variants in _HEADER_ALIASES.items():
        for alias in variants:
            alias_map[_normalise(alias)] = canonical
    return alias_map


_ALIAS_LOOKUP: Dict[str, str] = _build_lookup()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate_and_clean(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Normalise headers and coerce numeric columns.

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame with canonical column names.
    mapping    : Dict[str, str]
        {original_header → canonical_header}
    missing    : List[str]
        Any of the five required headers still missing after mapping.
    """
    original_cols = list(df.columns)
    mapping: Dict[str, str] = {}

    # ---- 1. Header mapping ------------------------------------------------- #
    renamed = {}
    for col in original_cols:
        key = _normalise(col)
        canonical = _ALIAS_LOOKUP.get(key)
        if canonical and canonical not in renamed.values():
            renamed[col] = canonical
            mapping[col] = canonical
    df = df.rename(columns=renamed)

    # ---- 2. Numeric coercion ---------------------------------------------- #
    numeric_cols = ["clicks", "impressions", "position"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 3. Final validation ---------------------------------------------- #
    required = set(_HEADER_ALIASES.keys())
    missing = [col for col in required if col not in df.columns]

    return df, mapping, missing
