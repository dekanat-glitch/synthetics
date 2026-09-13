
import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Media Partners Intelligence",
    page_icon="◉",
    layout="wide",
)

DATABASE_URL = os.environ.get("NEON_DATABASE_URL")

if not DATABASE_URL:
    try:
        DATABASE_URL = st.secrets["NEON_DATABASE_URL"]
    except Exception:
        DATABASE_URL = None

if not DATABASE_URL:
    st.error(
        "NEON_DATABASE_URL is missing. "
        "Add the demo Neon branch connection string in "
        "Streamlit Community Cloud → App settings → Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Database helpers
# ---------------------------------------------------------

@st.cache_data(ttl=60)
def list_tables():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            return {row[0] for row in cur.fetchall()}


@st.cache_data(ttl=60)
def load_table(table_name):
    allowed = list_tables()

    if table_name not in allowed:
        return pd.DataFrame()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT * FROM "{table_name}";'
            )
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    return pd.DataFrame(rows, columns=cols)


def safe_datetime(series):
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce"
    )


def maybe_url(row):
    for col in ["canonical_url", "web_url", "url"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return None


def article_url_column(df):
    result = pd.Series(
        [None] * len(df),
        index=df.index,
        dtype="object"
    )

    for col in ["canonical_url", "web_url"]:
        if col in df.columns:
            result = result.where(
                result.notna(),
                df[col]
            )

    return result


TABLES = list_tables()

media = load_table("media")
articles = load_table("articles")
topics = load_table("topics")
article_topics = load_table("article_topics")
entities = load_table("entities")
article_entities = load_table("article_entities")
citations_raw = load_table("citations")
social_posts = load_table("social_posts")
article_social_links = load_table("article_social_links")
story_clusters = load_table("story_clusters")
story_cluster_articles = load_table("story_cluster_articles")
article_relations = load_table("article_relations")
metrics_snapshots = load_table("metrics_snapshots")
demo_story_journeys = load_table("demo_story_journeys")


# ---------------------------------------------------------
# Normalize core data
# ---------------------------------------------------------

if articles.empty:
    st.error("No rows found in articles.")
    st.stop()

if "published_at" in articles.columns:
    articles["published_at"] = safe_datetime(
        articles["published_at"]
    )
else:
    articles["published_at"] = pd.NaT

articles["article_url"] = article_url_column(
    articles
)

if not media.empty and "media_id" in articles.columns:
    media_lookup = (
        media[["id", "name"]]
        .rename(
            columns={
                "id": "media_id",
                "name": "media"
            }
        )
    )

    articles = articles.merge(
        media_lookup,
        on="media_id",
        how="left"
    )

else:
    articles["media"] = "Unknown"


article_meta = articles[
    [
        c for c in [
            "id",
            "source_key",
            "media_id",
            "media",
            "title",
            "body_text",
            "language",
            "published_at",
            "article_url",
        ]
        if c in articles.columns
    ]
].copy()


# ---------------------------------------------------------
# Topic expansion
# ---------------------------------------------------------

topic_expanded = pd.DataFrame()

if (
    not article_topics.empty
    and not topics.empty
    and {"article_id", "topic_id"}.issubset(
        article_topics.columns
    )
):
    topic_cols = [
        c for c in [
            "id",
            "name",
            "topic_group"
        ]
        if c in topics.columns
    ]

    topic_lookup = topics[
        topic_cols
    ].rename(
        columns={
            "id": "topic_id",
            "name": "topic"
        }
    )

    topic_expanded = (
        article_topics
        .merge(
            topic_lookup,
            on="topic_id",
            how="left"
        )
        .merge(
            article_meta.rename(
                columns={"id": "article_id"}
            ),
            on="article_id",
            how="left"
        )
    )


# ---------------------------------------------------------
# Entity expansion
# ---------------------------------------------------------

entity_expanded = pd.DataFrame()

if (
    not article_entities.empty
    and not entities.empty
    and {"article_id", "entity_id"}.issubset(
        article_entities.columns
    )
):
    entity_cols = [
        c for c in [
            "id",
            "canonical_name",
            "entity_type"
        ]
        if c in entities.columns
    ]

    entity_lookup = entities[
        entity_cols
    ].rename(
        columns={
            "id": "entity_id",
            "canonical_name": "entity"
        }
    )

    entity_expanded = (
        article_entities
        .merge(
            entity_lookup,
            on="entity_id",
            how="left"
        )
        .merge(
            article_meta.rename(
                columns={"id": "article_id"}
            ),
            on="article_id",
            how="left"
        )
    )


# ---------------------------------------------------------
# Citation expansion
# ---------------------------------------------------------

citations = citations_raw.copy()

if not citations.empty:

    left_meta = article_meta[
        [
            c for c in [
                "id",
                "source_key",
                "media",
                "title",
                "published_at",
                "article_url"
            ]
            if c in article_meta.columns
        ]
    ].rename(
        columns={
            "id": "citing_article_id",
            "source_key": "citing_source_key",
            "media": "citing_media",
            "title": "citing_title",
            "published_at": "citing_published_at",
            "article_url": "citing_url",
        }
    )

    if "citing_article_id" in citations.columns:
        citations = citations.merge(
            left_meta,
            on="citing_article_id",
            how="left"
        )

    right_meta = article_meta[
        [
            c for c in [
                "id",
                "source_key",
                "media",
                "title",
                "published_at",
                "article_url"
            ]
            if c in article_meta.columns
        ]
    ].rename(
        columns={
            "id": "cited_article_id",
            "source_key": "cited_source_key",
            "media": "cited_media",
            "title": "cited_title",
            "published_at": "cited_published_at",
            "article_url": "cited_url",
        }
    )

    if "cited_article_id" in citations.columns:
        citations = citations.merge(
            right_meta,
            on="cited_article_id",
            how="left"
        )


# ---------------------------------------------------------
# Story expansion
# ---------------------------------------------------------

story_memberships = pd.DataFrame()

if (
    not story_clusters.empty
    and not story_cluster_articles.empty
    and {"cluster_id", "article_id"}.issubset(
        story_cluster_articles.columns
    )
):
    cluster_cols = [
        c for c in [
            "id",
            "source_key",
            "title",
            "summary",
            "status",
            "confidence",
            "first_published_at"
        ]
        if c in story_clusters.columns
    ]

    cluster_lookup = story_clusters[
        cluster_cols
    ].rename(
        columns={
            "id": "cluster_id",
            "source_key": "cluster_seed",
            "title": "cluster_title",
            "summary": "cluster_summary",
            "status": "cluster_status",
            "confidence": "cluster_confidence",
        }
    )

    story_memberships = (
        story_cluster_articles
        .merge(
            cluster_lookup,
            on="cluster_id",
            how="left"
        )
        .merge(
            article_meta.rename(
                columns={"id": "article_id"}
            ),
            on="article_id",
            how="left"
        )
    )

    if "first_published_at" in story_memberships.columns:
        story_memberships[
            "first_published_at"
        ] = safe_datetime(
            story_memberships[
                "first_published_at"
            ]
        )


# ---------------------------------------------------------
# Social-post normalization
# ---------------------------------------------------------

if not social_posts.empty:

    if "published_at" in social_posts.columns:
        social_posts["published_at"] = safe_datetime(
            social_posts["published_at"]
        )

    if (
        "media_id" in social_posts.columns
        and not media.empty
    ):
        social_posts = social_posts.merge(
            media[
                ["id", "name"]
            ].rename(
                columns={
                    "id": "media_id",
                    "name": "media"
                }
            ),
            on="media_id",
            how="left"
        )


def normalize_article_id_series(series):
    """
    Accept either Neon numeric article IDs or temporary
    source keys such as art_....
    """
    if series is None:
        return pd.Series(dtype="Int64")

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Int64"
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric_mask = numeric.notna()

    result.loc[numeric_mask] = (
        numeric.loc[numeric_mask]
        .astype("int64")
    )

    if "source_key" in article_meta.columns:

        key_map = dict(
            zip(
                article_meta[
                    "source_key"
                ].astype(str),
                article_meta["id"]
            )
        )

        text_mask = ~numeric_mask

        mapped = (
            series.loc[text_mask]
            .astype(str)
            .map(key_map)
        )

        valid = mapped.notna()

        result.loc[
            mapped.index[valid]
        ] = mapped.loc[valid].astype("int64")

    return result


social_link_rows = []

# Route 1: social_posts.article_id directly
if (
    not social_posts.empty
    and "article_id" in social_posts.columns
):

    normalized_article_ids = normalize_article_id_series(
        social_posts["article_id"]
    )

    for idx, row in social_posts.iterrows():
        article_id = normalized_article_ids.loc[idx]

        if pd.notna(article_id):
            social_link_rows.append({
                "article_id": int(article_id),
                "social_post_id": (
                    row["id"]
                    if "id" in social_posts.columns
                    else None
                ),
                "platform": (
                    row["platform"]
                    if "platform" in social_posts.columns
                    else None
                ),
                "post_url": (
                    row["post_url"]
                    if "post_url" in social_posts.columns
                    else None
                ),
                "post_text": (
                    row["post_text"]
                    if "post_text" in social_posts.columns
                    else None
                ),
                "published_at": (
                    row["published_at"]
                    if "published_at" in social_posts.columns
                    else pd.NaT
                ),
                "relation_method": (
                    row["article_relation_method"]
                    if "article_relation_method" in social_posts.columns
                    else None
                ),
                "relation_confidence": (
                    row["article_relation_confidence"]
                    if "article_relation_confidence" in social_posts.columns
                    else None
                ),
            })


# Route 2: separate article_social_links table
if not article_social_links.empty:

    article_col = None

    for candidate in [
        "article_id",
        "article_db_id",
        "article_source_key"
    ]:
        if candidate in article_social_links.columns:
            article_col = candidate
            break

    post_col = None

    for candidate in [
        "social_post_id",
        "post_id"
    ]:
        if candidate in article_social_links.columns:
            post_col = candidate
            break

    if article_col is not None:

        normalized_article_ids = normalize_article_id_series(
            article_social_links[
                article_col
            ]
        )

        social_post_lookup = {}

        if (
            post_col is not None
            and not social_posts.empty
            and "id" in social_posts.columns
        ):
            social_post_lookup = (
                social_posts
                .set_index("id")
                .to_dict("index")
            )

        for idx, link_row in (
            article_social_links.iterrows()
        ):
            article_id = normalized_article_ids.loc[idx]

            if pd.isna(article_id):
                continue

            post_id = (
                link_row[post_col]
                if post_col is not None
                else None
            )

            post_meta = {}

            if post_id in social_post_lookup:
                post_meta = social_post_lookup[
                    post_id
                ]

            social_link_rows.append({
                "article_id": int(article_id),
                "social_post_id": post_id,
                "platform": post_meta.get(
                    "platform"
                ),
                "post_url": post_meta.get(
                    "post_url"
                ),
                "post_text": post_meta.get(
                    "post_text"
                ),
                "published_at": post_meta.get(
                    "published_at",
                    pd.NaT
                ),
                "relation_method": (
                    link_row["relation_method"]
                    if "relation_method" in article_social_links.columns
                    else None
                ),
                "relation_confidence": (
                    link_row["relation_confidence"]
                    if "relation_confidence" in article_social_links.columns
                    else None
                ),
            })


social_links = pd.DataFrame(
    social_link_rows
).drop_duplicates()


# Human-readable website ↔ social table.
article_social_view = pd.DataFrame()

if not social_links.empty:

    article_social_view = (
        social_links
        .merge(
            article_meta[
                [
                    c for c in [
                        "id",
                        "media",
                        "title",
                        "published_at",
                        "article_url"
                    ]
                    if c in article_meta.columns
                ]
            ].rename(
                columns={
                    "id": "article_id",
                    "published_at": "article_published_at"
                }
            ),
            on="article_id",
            how="left"
        )
    )

    if "platform" in article_social_view.columns:
        article_social_view["platform"] = (
            article_social_view["platform"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )



# ---------------------------------------------------------
# Metrics normalization
# ---------------------------------------------------------

latest_metrics = pd.DataFrame()
web_latest = pd.DataFrame()
social_latest = pd.DataFrame()

if not metrics_snapshots.empty:

    if "snapshot_at" in metrics_snapshots.columns:
        metrics_snapshots["snapshot_at"] = safe_datetime(
            metrics_snapshots["snapshot_at"]
        )

    metric_numeric_cols = [
        "views_count",
        "reach_count",
        "users_count",
        "likes_count",
        "reactions_count",
        "comments_count",
        "shares_count",
        "forwards_count",
        "clicks_count",
        "engaged_sessions",
        "avg_engagement_seconds",
    ]

    for col in metric_numeric_cols:
        if col in metrics_snapshots.columns:
            metrics_snapshots[col] = pd.to_numeric(
                metrics_snapshots[col],
                errors="coerce"
            )

    # Metrics are cumulative snapshots. For totals, use only the
    # latest snapshot for each content item and platform.
    latest_metrics = (
        metrics_snapshots
        .sort_values("snapshot_at")
        .groupby(
            ["target_type", "target_id", "platform"],
            as_index=False,
            dropna=False
        )
        .tail(1)
        .copy()
    )

    web_latest = latest_metrics[
        latest_metrics["target_type"].eq("article")
    ].copy()

    if not web_latest.empty:
        web_latest = web_latest.merge(
            article_meta[
                [
                    c for c in [
                        "id",
                        "media",
                        "title",
                        "published_at",
                        "article_url"
                    ]
                    if c in article_meta.columns
                ]
            ].rename(columns={"id": "target_id"}),
            on="target_id",
            how="left"
        )
        web_latest = web_latest.rename(
            columns={"target_id": "article_id"}
        )

    social_latest = latest_metrics[
        latest_metrics["target_type"].eq("social_post")
    ].copy()

    if (
        not social_latest.empty
        and not social_posts.empty
        and "id" in social_posts.columns
    ):
        social_cols = [
            c for c in [
                "id",
                "article_id",
                "media",
                "platform",
                "published_at",
                "post_url",
                "post_text"
            ]
            if c in social_posts.columns
        ]

        social_latest = social_latest.merge(
            social_posts[social_cols].rename(
                columns={
                    "id": "target_id",
                    "published_at": "post_published_at",
                    "platform": "post_platform"
                }
            ),
            on="target_id",
            how="left"
        )

        # Prefer the platform attached to the social post.
        if "post_platform" in social_latest.columns:
            social_latest["platform"] = (
                social_latest["post_platform"]
                .fillna(social_latest["platform"])
                .astype(str)
                .str.lower()
            )

        if "article_id" in social_latest.columns:
            social_latest["article_id"] = normalize_article_id_series(
                social_latest["article_id"]
            )

            social_latest = social_latest.merge(
                article_meta[
                    [
                        c for c in [
                            "id",
                            "title",
                            "published_at",
                            "article_url"
                        ]
                        if c in article_meta.columns
                    ]
                ].rename(
                    columns={
                        "id": "article_id",
                        "title": "article_title",
                        "published_at": "article_published_at"
                    }
                ),
                on="article_id",
                how="left"
            )


def sum_metric(df, col):
    if df.empty or col not in df.columns:
        return 0
    return float(
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0)
        .sum()
    )


def compact_number(value):
    value = float(value or 0)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{int(round(value)):,}"


def social_interactions(df):
    if df.empty:
        return 0

    cols = [
        c for c in [
            "likes_count",
            "reactions_count",
            "comments_count",
            "shares_count",
            "forwards_count",
        ]
        if c in df.columns
    ]

    if not cols:
        return 0

    return float(
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        .sum()
    )


# ---------------------------------------------------------
# Article-level performance + metadata maps
# ---------------------------------------------------------


def join_unique(values, limit=5):
    clean = []
    seen = set()

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text or text in seen:
            continue

        clean.append(text)
        seen.add(text)

    suffix = ""

    if len(clean) > limit:
        suffix = f" +{len(clean) - limit}"

    return ", ".join(clean[:limit]) + suffix


article_topic_map = {}
article_people_map = {}
article_other_entities_map = {}

if not topic_expanded.empty:
    topic_tmp = topic_expanded.copy()

    if "role" in topic_tmp.columns:
        role_order = {
            "primary": 0,
            "secondary": 1,
        }
        topic_tmp["_role_order"] = (
            topic_tmp["role"]
            .map(role_order)
            .fillna(2)
        )
    else:
        topic_tmp["_role_order"] = 2

    if "rank" not in topic_tmp.columns:
        topic_tmp["rank"] = 999

    topic_tmp = topic_tmp.sort_values(
        ["article_id", "_role_order", "rank"]
    )

    article_topic_map = (
        topic_tmp
        .groupby("article_id")["topic"]
        .apply(lambda s: join_unique(s, limit=6))
        .to_dict()
    )

if not entity_expanded.empty:
    people_tmp = entity_expanded[
        entity_expanded["entity_type"].eq("person")
    ].copy()

    other_tmp = entity_expanded[
        ~entity_expanded["entity_type"].eq("person")
    ].copy()

    if not people_tmp.empty:
        article_people_map = (
            people_tmp
            .groupby("article_id")["entity"]
            .apply(lambda s: join_unique(s, limit=5))
            .to_dict()
        )

    if not other_tmp.empty:
        article_other_entities_map = (
            other_tmp
            .groupby("article_id")["entity"]
            .apply(lambda s: join_unique(s, limit=5))
            .to_dict()
        )


article_performance = article_meta[
    [
        c for c in [
            "id",
            "media",
            "title",
            "published_at",
            "article_url",
        ]
        if c in article_meta.columns
    ]
].rename(columns={"id": "article_id"}).copy()

if not web_latest.empty:
    web_agg = (
        web_latest
        .groupby("article_id", as_index=False)
        .agg(
            website_views=("views_count", "sum"),
            website_users=("users_count", "sum"),
            avg_engagement_seconds=("avg_engagement_seconds", "mean"),
        )
    )

    article_performance = article_performance.merge(
        web_agg,
        on="article_id",
        how="left"
    )

if not social_latest.empty and "article_id" in social_latest.columns:
    social_perf_tmp = social_latest.copy()

    for col in [
        "views_count",
        "reach_count",
        "likes_count",
        "reactions_count",
        "comments_count",
        "shares_count",
        "forwards_count",
        "clicks_count",
    ]:
        if col not in social_perf_tmp.columns:
            social_perf_tmp[col] = 0

    social_perf_tmp["interactions"] = (
        social_perf_tmp[
            [
                "likes_count",
                "reactions_count",
                "comments_count",
                "shares_count",
                "forwards_count",
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )

    social_agg = (
        social_perf_tmp
        .dropna(subset=["article_id"])
        .groupby("article_id", as_index=False)
        .agg(
            social_posts=("target_id", "nunique"),
            social_views=("views_count", "sum"),
            social_reach=("reach_count", "sum"),
            social_clicks=("clicks_count", "sum"),
            social_interactions=("interactions", "sum"),
        )
    )

    article_performance = article_performance.merge(
        social_agg,
        on="article_id",
        how="left"
    )

for col in [
    "website_views",
    "website_users",
    "avg_engagement_seconds",
    "social_posts",
    "social_views",
    "social_reach",
    "social_clicks",
    "social_interactions",
]:
    if col not in article_performance.columns:
        article_performance[col] = 0

    article_performance[col] = pd.to_numeric(
        article_performance[col],
        errors="coerce"
    ).fillna(0)

article_performance["observed_distribution"] = (
    article_performance["website_views"]
    + article_performance["social_reach"]
)

article_perf_lookup = (
    article_performance
    .set_index("article_id")
    .to_dict("index")
    if not article_performance.empty
    else {}
)


def article_metrics(article_id):
    return article_perf_lookup.get(
        article_id,
        {
            "website_views": 0,
            "social_views": 0,
            "social_reach": 0,
            "social_posts": 0,
            "social_clicks": 0,
            "social_interactions": 0,
            "observed_distribution": 0,
        }
    )


def associated_performance(article_ids):
    ids = set(article_ids)

    if not ids or article_performance.empty:
        return {
            "website_views": 0,
            "social_reach": 0,
            "social_views": 0,
            "observed_distribution": 0,
        }

    tmp = article_performance[
        article_performance["article_id"].isin(ids)
    ]

    return {
        "website_views": sum_metric(tmp, "website_views"),
        "social_reach": sum_metric(tmp, "social_reach"),
        "social_views": sum_metric(tmp, "social_views"),
        "observed_distribution": sum_metric(tmp, "observed_distribution"),
    }


# ---------------------------------------------------------
# Global filters
# ---------------------------------------------------------

st.title("Media Partners Intelligence")
st.caption(
    "Partner publishing, topics, people, citations, story movement, "
    "website ↔ social distribution and content performance."
)
st.info(
    "DEMO ENVIRONMENT · Synthetic data · August 2026 · "
    "Partner 1, Partner 2, Partner 3"
)
st.caption(
    "Content → Distribution → Impact · confirmed evidence is kept separate "
    "from AI/inference-based relations."
)

with st.sidebar:

    st.header("Global filters")

    media_options = sorted(
        [
            x
            for x in articles[
                "media"
            ].dropna().unique()
        ]
    )

    selected_media = st.multiselect(
        "Partners",
        options=media_options,
        default=media_options
    )

    valid_dates = articles[
        "published_at"
    ].dropna()

    if len(valid_dates):
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.date_input(
            "Publication date",
            value=(
                min_date,
                max_date
            ),
            min_value=min_date,
            max_value=max_date
        )

    else:
        selected_dates = None

    st.divider()

    if st.button(
        "Refresh Neon data",
        use_container_width=True
    ):
        st.cache_data.clear()
        st.rerun()


filtered_articles = articles.copy()

if selected_media:
    filtered_articles = (
        filtered_articles[
            filtered_articles[
                "media"
            ].isin(
                selected_media
            )
        ]
    )

if (
    selected_dates
    and isinstance(
        selected_dates,
        (tuple, list)
    )
    and len(selected_dates) == 2
):

    start_date, end_date = selected_dates

    filtered_articles = (
        filtered_articles[
            filtered_articles[
                "published_at"
            ].dt.date.between(
                start_date,
                end_date
            )
        ]
    )


filtered_article_ids = set(
    filtered_articles["id"]
)


filtered_social_posts = social_posts.copy()

if not filtered_social_posts.empty:

    if selected_media and "media" in filtered_social_posts.columns:
        filtered_social_posts = filtered_social_posts[
            filtered_social_posts["media"].isin(selected_media)
        ]

    if (
        selected_dates
        and isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
        and "published_at" in filtered_social_posts.columns
    ):
        start_date, end_date = selected_dates

        filtered_social_posts = filtered_social_posts[
            filtered_social_posts["published_at"]
            .dt.date
            .between(start_date, end_date)
        ]


# ---------------------------------------------------------
# Utility renderers
# ---------------------------------------------------------

def article_table(df, limit=200):
    if df.empty:
        st.info("No publications match these filters.")
        return

    show = df.copy()

    cols = [
        c for c in [
            "published_at",
            "media",
            "title",
            "article_url"
        ]
        if c in show.columns
    ]

    show = show[
        cols
    ].sort_values(
        "published_at",
        ascending=False
    ).head(limit)

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "published_at": st.column_config.DatetimeColumn(
                "Published",
                format="YYYY-MM-DD HH:mm"
            ),
            "media": "Partner",
            "title": "Publication",
            "article_url": st.column_config.LinkColumn(
                "Open"
            ),
        },
    )


def metric_number(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)




# ---------------------------------------------------------
# V7 visualization + analysis helpers
# ---------------------------------------------------------

PARTNER_PALETTE = (
    px.colors.qualitative.Dark24
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Bold
)


def partner_color_map(partners):
    names = sorted([str(x) for x in partners if pd.notna(x)])
    return {
        name: PARTNER_PALETTE[i % len(PARTNER_PALETTE)]
        for i, name in enumerate(names)
    }


def _hex_to_rgb(hex_color):
    value = str(hex_color).lstrip('#')
    if len(value) != 6:
        return (80, 110, 160)
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(
        max(0, min(255, int(v))) for v in rgb
    )


def lighten_color(hex_color, amount=0.25):
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex((
        r + (255-r) * amount,
        g + (255-g) * amount,
        b + (255-b) * amount,
    ))


def platform_color(base_color, platform):
    p = str(platform or '').lower()
    amount = {
        'website': 0.00,
        'web': 0.00,
        'telegram': 0.20,
        'facebook': 0.42,
        'instagram': 0.58,
    }.get(p, 0.30)
    return lighten_color(base_color, amount)


def format_hours(value):
    if value is None or pd.isna(value):
        return '---'
    value = float(value)
    if value < 1:
        return f'{value*60:.0f} min'
    return f'{value:.1f} h'


def citation_view_for(article_ids=None):
    if citations.empty:
        return pd.DataFrame()

    cv = citations.copy()
    if article_ids is not None:
        ids = set(article_ids)
        mask = pd.Series(False, index=cv.index)
        if 'citing_article_id' in cv.columns:
            mask = mask | cv['citing_article_id'].isin(ids)
        if 'cited_article_id' in cv.columns:
            mask = mask | cv['cited_article_id'].isin(ids)
        cv = cv[mask].copy()

    for side in ['citing', 'cited']:
        id_col = f'{side}_article_id'
        if id_col not in cv.columns:
            continue
        cv[f'{side}_topics'] = cv[id_col].map(article_topic_map).fillna('')
        cv[f'{side}_people'] = cv[id_col].map(article_people_map).fillna('')
        cv[f'{side}_entities'] = cv[id_col].map(article_other_entities_map).fillna('')
        cv[f'{side}_web_views'] = cv[id_col].map(
            lambda x: article_metrics(x).get('website_views', 0)
        )
        cv[f'{side}_social_reach'] = cv[id_col].map(
            lambda x: article_metrics(x).get('social_reach', 0)
        )

    return cv


def citation_flow_table(cv):
    if cv.empty or not {'citing_media', 'cited_media'}.issubset(cv.columns):
        return pd.DataFrame(columns=['Source', 'Target', 'Count'])

    resolved = cv.dropna(subset=['citing_media', 'cited_media']).copy()
    resolved = resolved[resolved['citing_media'] != resolved['cited_media']]
    if resolved.empty:
        return pd.DataFrame(columns=['Source', 'Target', 'Count'])

    # Direction is information flow: cited/source -> citing/receiver.
    return (
        resolved
        .groupby(['cited_media', 'citing_media'])
        .size()
        .reset_index(name='Count')
        .rename(columns={'cited_media': 'Source', 'citing_media': 'Target'})
    )


def directional_network_figure(flows, focus_partner=None, title=None):
    """Directed partner network with clearly separated curved reciprocal arrows.

    Source -> Target means information flow from the cited/source partner
    to the partner that cited it. When both directions exist between the
    same pair, the two arrows bend to opposite sides so they never overlap.
    """
    if flows.empty:
        return None

    data = flows.copy()
    if focus_partner and focus_partner != 'All partners':
        data = data[
            data['Source'].eq(focus_partner)
            | data['Target'].eq(focus_partner)
        ].copy()

    if data.empty:
        return None

    nodes = sorted(set(data['Source']) | set(data['Target']))
    colors = partner_color_map(nodes)
    n = len(nodes)

    # --- Node layout -------------------------------------------------
    # For overview: circle. For focused partner: center + neighbours around it.
    positions = {}
    if n == 1:
        positions[nodes[0]] = (0.0, 0.0)
    elif focus_partner and focus_partner != 'All partners' and focus_partner in nodes:
        positions[focus_partner] = (0.0, 0.0)
        others = [x for x in nodes if x != focus_partner]
        for i, node in enumerate(others):
            angle = math.pi / 2 - 2 * math.pi * i / max(len(others), 1)
            positions[node] = (1.05 * math.cos(angle), 1.05 * math.sin(angle))
    else:
        for i, node in enumerate(nodes):
            angle = math.pi / 2 - 2 * math.pi * i / n
            positions[node] = (math.cos(angle), math.sin(angle))

    # One row per directed relation. Reciprocal pairs occur twice.
    directed_pairs = set(zip(data['Source'], data['Target']))

    def quadratic_bezier(p0, p1, bend, points=60):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        # perpendicular unit vector
        nx, ny = -dy / length, dx / length
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        cx, cy = mx + nx * bend, my + ny * bend

        xs, ys = [], []
        for j in range(points):
            t = j / (points - 1)
            omt = 1 - t
            xs.append(omt * omt * x0 + 2 * omt * t * cx + t * t * x1)
            ys.append(omt * omt * y0 + 2 * omt * t * cy + t * t * y1)

        # label at midpoint
        t = 0.5
        omt = 1 - t
        lx = omt * omt * x0 + 2 * omt * t * cx + t * t * x1
        ly = omt * omt * y0 + 2 * omt * t * cy + t * t * y1

        # Arrow-head direction near target
        t0, t1 = 0.88, 0.975
        def q(t):
            omt = 1 - t
            return (
                omt * omt * x0 + 2 * omt * t * cx + t * t * x1,
                omt * omt * y0 + 2 * omt * t * cy + t * t * y1,
            )
        ax, ay = q(t0)
        bx, by = q(t1)
        return xs, ys, (lx, ly), (ax, ay, bx, by)

    fig = go.Figure()

    for _, row in data.iterrows():
        source = str(row['Source'])
        target = str(row['Target'])
        if source == target:
            continue

        count = int(row['Count'])
        x0, y0 = positions[source]
        x1, y1 = positions[target]

        # If A->B and B->A both exist, bend to opposite sides.
        reciprocal = (target, source) in directed_pairs
        if reciprocal:
            # IMPORTANT: use the SAME bend sign for both directions.
            # The perpendicular vector flips automatically when source/target reverse,
            # so the two reciprocal arrows land on opposite sides of the straight line.
            bend = 0.42
        else:
            # Slight curve even for one-way links so all links share the same visual language.
            bend = 0.12

        xs, ys, (lx, ly), (ax, ay, bx, by) = quadratic_bezier(
            (x0, y0), (x1, y1), bend
        )

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(width=2.4, color='#7f8899'),
            hovertemplate=(
                f'<b>{source}</b> → <b>{target}</b>'
                f'<br>Confirmed citations: {count}<extra></extra>'
            ),
            showlegend=False,
        ))

        # Arrowhead follows the curve into the target.
        fig.add_annotation(
            x=bx, y=by,
            ax=ax, ay=ay,
            xref='x', yref='y', axref='x', ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.15,
            arrowwidth=2.3,
            arrowcolor='#7f8899',
        )

        # Count sits on its own arc.
        fig.add_annotation(
            x=lx,
            y=ly,
            text=f'<b>{count}</b>',
            showarrow=False,
            bgcolor='white',
            bordercolor='#d8dde6',
            borderwidth=1,
            borderpad=4,
            font=dict(size=14, color='#17223b'),
        )

    # Nodes sit above edges.
    node_sizes = []
    for node in nodes:
        total = int(
            data.loc[data['Source'].eq(node), 'Count'].sum()
            + data.loc[data['Target'].eq(node), 'Count'].sum()
        )
        base = 52 if node == focus_partner else 46
        node_sizes.append(base + min(16, total * 0.7))

    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in nodes],
        y=[positions[n][1] for n in nodes],
        mode='markers+text',
        text=[f'<b>{n}</b>' for n in nodes],
        textposition='bottom center',
        textfont=dict(
            color='#1f2937',
            size=14,
            family='Arial, sans-serif',
        ),
        marker=dict(
            size=node_sizes,
            color=[colors[n] for n in nodes],
            line=dict(color='white', width=3),
        ),
        customdata=nodes,
        hovertemplate='<b>%{customdata}</b><extra></extra>',
        showlegend=False,
    ))

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, scaleanchor='x', scaleratio=1)
    fig.update_layout(
        height=max(540, min(780, 460 + 14 * n)),
        margin=dict(l=30, r=30, t=55 if title else 25, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
    )
    if title:
        fig.update_layout(title=dict(text=str(title), x=0.02, xanchor='left'))

    # Padding around the network so labels and arcs do not clip.
    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    if xs and ys:
        x_span = max(max(xs) - min(xs), 1.0)
        y_span = max(max(ys) - min(ys), 1.0)
        fig.update_xaxes(range=[min(xs) - 0.30 * x_span, max(xs) + 0.30 * x_span])
        fig.update_yaxes(range=[min(ys) - 0.34 * y_span, max(ys) + 0.34 * y_span])

    return fig

def build_story_journey(selected_story):
    if selected_story.empty:
        return pd.DataFrame(), pd.DataFrame(), set()

    article_ids = set(selected_story['article_id'].dropna())
    social = (
        social_latest[social_latest['article_id'].isin(article_ids)].copy()
        if not social_latest.empty and 'article_id' in social_latest.columns
        else pd.DataFrame()
    )

    rows = []
    for _, row in selected_story.sort_values('published_at').iterrows():
        metrics = article_metrics(row['article_id'])
        rows.append({
            'event_id': f"article-{row['article_id']}",
            'article_id': row['article_id'],
            'Published': row.get('published_at'),
            'Partner': row.get('media'),
            'Channel': 'Website',
            'Type': 'Article',
            'Content': row.get('title'),
            'Views': metrics.get('website_views', 0),
            'Reach': np.nan,
            'Audience': metrics.get('website_views', 0),
            'Clicks': np.nan,
            'URL': row.get('article_url'),
        })

    if not social.empty:
        for _, row in social.iterrows():
            reach = pd.to_numeric(pd.Series([row.get('reach_count')]), errors='coerce').fillna(0).iloc[0]
            views = pd.to_numeric(pd.Series([row.get('views_count')]), errors='coerce').fillna(0).iloc[0]
            audience = reach if reach > 0 else views
            rows.append({
                'event_id': f"social-{row.get('target_id')}",
                'article_id': row.get('article_id'),
                'Published': row.get('post_published_at'),
                'Partner': row.get('media'),
                'Channel': str(row.get('platform', 'social')).title(),
                'Type': 'Social post',
                'Content': row.get('post_text'),
                'Views': views,
                'Reach': reach,
                'Audience': audience,
                'Clicks': row.get('clicks_count', 0),
                'URL': row.get('post_url'),
            })

    journey = pd.DataFrame(rows)
    if not journey.empty:
        journey['Published'] = pd.to_datetime(journey['Published'], utc=True, errors='coerce')
        journey = journey.sort_values('Published').reset_index(drop=True)
    return journey, social, article_ids


def story_distribution_figure(journey_df, story_citations):
    if journey_df.empty:
        return None

    df = journey_df.dropna(subset=['Published', 'Partner']).copy().reset_index(drop=True)
    if df.empty:
        return None

    partners = sorted(df['Partner'].dropna().unique())
    pcolors = partner_color_map(partners)

    audience = pd.to_numeric(df['Audience'], errors='coerce').fillna(0).clip(lower=1)
    amin, amax = float(audience.min()), float(audience.max())
    if amax <= amin:
        sizes = np.repeat(42.0, len(df))
    else:
        scaled = (np.sqrt(audience) - np.sqrt(amin)) / (np.sqrt(amax) - np.sqrt(amin))
        sizes = 30 + 38 * scaled
    df['node_size'] = sizes
    df['node_color'] = [
        platform_color(pcolors.get(str(r['Partner']), '#4c78a8'), r['Channel'])
        for _, r in df.iterrows()
    ]

    fig = go.Figure()

    # Gray chronological path across all publication events.
    for i in range(len(df)-1):
        x0 = df.loc[i, 'Published']
        x1 = df.loc[i+1, 'Published']
        fig.add_annotation(
            x=x1, y=0,
            ax=x0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            text='', showarrow=True, arrowhead=2, arrowsize=1.0,
            arrowwidth=1.5, arrowcolor='#a3aab7', opacity=0.85,
        )

    # Confirmed citations are green arcs from cited/source website article to citing website article.
    article_x = {
        int(r['article_id']): r['Published']
        for _, r in df[df['Type'].eq('Article')].iterrows()
        if pd.notna(r.get('article_id'))
    }
    if not story_citations.empty and {'citing_article_id', 'cited_article_id'}.issubset(story_citations.columns):
        for _, row in story_citations.dropna(subset=['citing_article_id', 'cited_article_id']).iterrows():
            source_id = int(row['cited_article_id'])
            target_id = int(row['citing_article_id'])
            if source_id not in article_x or target_id not in article_x:
                continue
            x0 = article_x[source_id]
            x1 = article_x[target_id]
            if x1 <= x0:
                continue
            mid = x0 + (x1-x0)/2
            fig.add_trace(go.Scatter(
                x=[x0, mid, x1],
                y=[0.05, 0.30, 0.05],
                mode='lines',
                line=dict(color='#159447', width=3, shape='spline'),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=mid, y=0.33, text='Direct citation', showarrow=False,
                font=dict(color='#0e7a37', size=12),
                bgcolor='rgba(232,248,238,0.95)',
                bordercolor='#7bcf98', borderwidth=1, borderpad=3,
            )

    # Nodes, one common horizontal line. Labels alternate above/below for readability.
    channel_short = {'Website': 'WEB', 'Telegram': 'TG', 'Facebook': 'FB', 'Instagram': 'IG'}
    for i, row in df.iterrows():
        y_label = -0.22 if i % 2 == 0 else -0.34
        metric = row['Reach'] if pd.notna(row['Reach']) and float(row['Reach']) > 0 else row['Views']
        metric_name = 'reach' if pd.notna(row['Reach']) and float(row['Reach']) > 0 else 'views'
        label = (
            f"<b>{row['Partner']}</b><br>"
            f"{row['Channel']}<br>"
            f"{compact_number(metric)} {metric_name}"
        )
        fig.add_trace(go.Scatter(
            x=[row['Published']], y=[0], mode='markers+text',
            text=[channel_short.get(row['Channel'], str(row['Channel'])[:3].upper())],
            textposition='middle center',
            textfont=dict(color='white', size=11),
            marker=dict(
                size=float(row['node_size']), color=row['node_color'],
                line=dict(color='white', width=2),
            ),
            customdata=[[row['Partner'], row['Channel'], row['Content'], row['Views'], row['Reach']]],
            hovertemplate=(
                '<b>%{customdata[0]} - %{customdata[1]}</b><br>'
                '%{x|%Y-%m-%d %H:%M}<br>%{customdata[2]}<br>'
                'Views: %{customdata[3]:,.0f}<br>Reach: %{customdata[4]:,.0f}<extra></extra>'
            ),
            showlegend=False,
        ))
        fig.add_annotation(
            x=row['Published'], y=y_label, text=label,
            showarrow=False, align='center', font=dict(size=11, color='#17223b')
        )

    # Legend annotations for partner colors.
    for j, partner in enumerate(partners):
        fig.add_annotation(
            xref='paper', yref='paper', x=0.02 + j*0.16, y=1.05,
            text=f"<span style='color:{pcolors[partner]}'>●</span> {partner}",
            showarrow=False, font=dict(size=12), align='left'
        )

    fig.add_annotation(
        xref='paper', yref='paper', x=0.99, y=1.05,
        text='<span style="color:#159447">green = citation</span> | gray = distribution path | node size = audience',
        showarrow=False, xanchor='right', font=dict(size=11, color='#697386')
    )

    fig.update_yaxes(visible=False, range=[-0.50, 0.42], fixedrange=True)
    fig.update_xaxes(title='Time', showgrid=False)
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=70, b=35),
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='closest',
    )
    return fig


def partner_profile_dataframe(article_scope):
    partners = sorted(article_scope['media'].dropna().unique()) if not article_scope.empty else []
    id_to_media = article_meta.set_index('id')['media'].to_dict() if not article_meta.empty else {}
    rows = []

    citation_scope = citation_view_for(set(article_scope['id']))

    for partner in partners:
        pa = article_scope[article_scope['media'].eq(partner)].copy()
        pids = set(pa['id'])
        pp = article_performance[article_performance['article_id'].isin(pids)]
        linked = set(social_links[social_links['article_id'].isin(pids)]['article_id']) if not social_links.empty else set()

        cited_by = 0
        cites_others = 0
        citation_hours = []
        if not citation_scope.empty:
            rec = citation_scope[
                citation_scope['cited_media'].eq(partner)
                & citation_scope['citing_media'].ne(partner)
            ]
            made = citation_scope[
                citation_scope['citing_media'].eq(partner)
                & citation_scope['cited_media'].ne(partner)
            ]
            cited_by = len(rec)
            cites_others = len(made)
            if {'citing_published_at', 'cited_published_at'}.issubset(made.columns):
                delta = (
                    pd.to_datetime(made['citing_published_at'], utc=True, errors='coerce')
                    - pd.to_datetime(made['cited_published_at'], utc=True, errors='coerce')
                ).dt.total_seconds() / 3600
                citation_hours = delta[delta >= 0].dropna().tolist()

        shared = 0
        story_response_hours = []
        if not story_memberships.empty:
            memberships = story_memberships[story_memberships['article_id'].isin(pids)]
            for cid in memberships['cluster_id'].dropna().unique():
                cluster = story_memberships[story_memberships['cluster_id'].eq(cid)].dropna(subset=['published_at'])
                if cluster['media'].nunique() < 2:
                    continue
                shared += 1
                first_all = cluster['published_at'].min()
                partner_first = cluster[cluster['media'].eq(partner)]['published_at'].min()
                if pd.notna(first_all) and pd.notna(partner_first):
                    hours = (partner_first-first_all).total_seconds()/3600
                    if hours >= 0:
                        story_response_hours.append(hours)

        rows.append({
            'Partner': partner,
            'Articles cited by partners': cited_by,
            'References to partner publications': cites_others,
            'Shared stories': shared,
            'Median story response (h)': float(np.median(story_response_hours)) if story_response_hours else np.nan,
            'Median citation response (h)': float(np.median(citation_hours)) if citation_hours else np.nan,
            'Website views': sum_metric(pp, 'website_views'),
            'Social reach': sum_metric(pp, 'social_reach'),
            'Website to social coverage (%)': 100 * len(linked) / max(len(pa), 1),
        })

    return pd.DataFrame(rows)


def contribution_heatmap(profile, selected_partner):
    if profile.empty:
        return None

    metrics = [
        'Articles cited by partners',
        'References to partner publications',
        'Shared stories',
        'Median story response (h)',
        'Median citation response (h)',
        'Social reach',
        'Website to social coverage (%)',
    ]
    score = pd.DataFrame(index=profile.index)
    display = pd.DataFrame(index=profile.index)

    for metric in metrics:
        vals = pd.to_numeric(profile[metric], errors='coerce')
        if metric.startswith('Median'):
            valid = vals.dropna()
            if len(valid) <= 1 or valid.max() == valid.min():
                norm = pd.Series(0.5, index=vals.index)
            else:
                norm = 1 - (vals-valid.min())/(valid.max()-valid.min())
                norm = norm.fillna(0)
            display[metric] = vals.map(format_hours)
        else:
            valid = vals.dropna()
            if len(valid) <= 1 or valid.max() == valid.min():
                norm = pd.Series(0.5, index=vals.index)
            else:
                norm = (vals-valid.min())/(valid.max()-valid.min())
            norm = norm.fillna(0)
            if metric == 'Social reach':
                display[metric] = vals.map(compact_number)
            elif metric.endswith('(%)'):
                display[metric] = vals.map(lambda x: f'{x:.0f}%')
            else:
                display[metric] = vals.fillna(0).map(lambda x: f'{int(x)}')
        score[metric] = norm

    order = profile['Partner'].tolist()
    if selected_partner in order:
        order = [selected_partner] + [p for p in order if p != selected_partner]
    idx_map = {p: i for i, p in enumerate(profile['Partner'])}
    row_idx = [idx_map[p] for p in order]

    z = score.loc[row_idx, metrics].to_numpy()
    text = display.loc[row_idx, metrics].to_numpy()
    ylabels = [f'{p}  (selected)' if p == selected_partner else p for p in order]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=ylabels,
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        zmin=0, zmax=1,
        colorbar=dict(title='Relative<br>position'),
        hovertemplate='<b>%{y}</b><br>%{x}: %{text}<extra></extra>',
    ))
    fig.update_layout(
        height=max(330, 45*len(order)+150),
        margin=dict(l=20, r=20, t=25, b=120),
        xaxis=dict(tickangle=-28),
    )
    return fig


def theme_stats(article_ids, label, item_type):
    ids = set(article_ids)
    if not ids:
        return None
    subset = filtered_articles[filtered_articles['id'].isin(ids)]
    perf = associated_performance(ids)
    clusters = (
        story_memberships[story_memberships['article_id'].isin(ids)]['cluster_id'].nunique()
        if not story_memberships.empty else 0
    )
    cv = citation_view_for(ids)
    cited = (
        int(cv['cited_article_id'].isin(ids).sum())
        if not cv.empty and 'cited_article_id' in cv.columns else 0
    )
    rel = (
        article_relations[article_relations['source_article_id'].isin(ids)]
        if not article_relations.empty and 'source_article_id' in article_relations.columns
        else pd.DataFrame()
    )
    return {
        'Theme': label,
        'Type': item_type,
        'Articles': len(ids),
        'Partners': subset['media'].nunique(),
        'Shared stories': clusters,
        'Citations received': cited,
        'Possible pickups': int(rel['relation_type'].eq('possible_pickup').sum()) if not rel.empty else 0,
        'Follow-ups': int(rel['relation_type'].eq('follow_up').sum()) if not rel.empty else 0,
        'Website views': perf['website_views'],
        'Social reach': perf['social_reach'],
    }


# ---------------------------------------------------------
# Tabs - V7 information architecture
# ---------------------------------------------------------

tabs = st.tabs([
    'Overview',
    'Network Patterns',
    'Partner Perspective',
    'Citations',
    'Story Journeys',
    'Platform Distribution',
    'Publications Explorer',
    'Themes',
])


# =========================================================
# 1. OVERVIEW
# =========================================================

with tabs[0]:
    st.subheader('Overview')

    perf_scope = article_performance[
        article_performance['article_id'].isin(filtered_article_ids)
    ].copy()
    social_scope = (
        social_latest[social_latest['article_id'].isin(filtered_article_ids)].copy()
        if not social_latest.empty and 'article_id' in social_latest.columns
        else pd.DataFrame()
    )

    linked_ids = set(
        social_links[social_links['article_id'].isin(filtered_article_ids)]['article_id']
    ) if not social_links.empty else set()

    cv_scope = citation_view_for(filtered_article_ids)
    resolved_cv = (
        cv_scope.dropna(subset=['citing_media', 'cited_media'])
        if not cv_scope.empty and {'citing_media','cited_media'}.issubset(cv_scope.columns)
        else pd.DataFrame()
    )
    story_scope = (
        story_memberships[story_memberships['article_id'].isin(filtered_article_ids)]
        if not story_memberships.empty else pd.DataFrame()
    )
    rel_scope = (
        article_relations[
            article_relations['source_article_id'].isin(filtered_article_ids)
            | article_relations['related_article_id'].isin(filtered_article_ids)
        ]
        if not article_relations.empty and {'source_article_id','related_article_id'}.issubset(article_relations.columns)
        else pd.DataFrame()
    )

    st.markdown('#### Performance')
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric('Website views', compact_number(sum_metric(perf_scope, 'website_views')))
    k2.metric('Social views', compact_number(sum_metric(perf_scope, 'social_views')))
    k3.metric('Social reach', compact_number(sum_metric(perf_scope, 'social_reach')))
    k4.metric('Interactions', compact_number(sum_metric(perf_scope, 'social_interactions')))
    k5.metric('Clicks', compact_number(sum_metric(perf_scope, 'social_clicks')))
    avg_eng = pd.to_numeric(perf_scope['avg_engagement_seconds'], errors='coerce').replace(0, np.nan).mean() if not perf_scope.empty else np.nan
    k6.metric('Avg web engagement', f'{avg_eng:.0f}s' if pd.notna(avg_eng) else '---')

    st.caption('Social reach is summed across platform snapshots and is not deduplicated unique audience.')

    st.markdown('#### Network activity')
    n1,n2,n3,n4,n5,n6 = st.columns(6)
    n1.metric('Website articles', len(filtered_articles))
    n2.metric('Social posts', len(filtered_social_posts))
    n3.metric('Website to social coverage', f'{100*len(linked_ids)/max(len(filtered_articles),1):.0f}%')
    n4.metric('Direct citations', len(resolved_cv))
    n5.metric('Shared stories', story_scope['cluster_id'].nunique() if not story_scope.empty else 0)
    n6.metric('Possible pickups / follow-ups', len(rel_scope))

    st.markdown('#### Audience by partner')
    by_partner = (
        perf_scope.groupby('media', as_index=False)
        .agg(website_views=('website_views','sum'), social_reach=('social_reach','sum'))
        if not perf_scope.empty else pd.DataFrame()
    )
    if not by_partner.empty:
        long = by_partner.melt(id_vars='media', var_name='Metric', value_name='Value')
        long['Metric'] = long['Metric'].map({'website_views':'Website views','social_reach':'Social reach'})
        fig = px.bar(long, y='media', x='Value', color='Metric', barmode='group', orientation='h',
                     labels={'media':'Partner','Value':'Audience'})
        fig.update_layout(height=max(330, 60*by_partner['media'].nunique()+100), margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True, key='overview_audience_by_partner')

    st.markdown('#### Website to platform coverage')
    coverage_rows=[]
    for platform in ['telegram','facebook','instagram']:
        linked = (
            article_social_view[
                article_social_view['article_id'].isin(filtered_article_ids)
                & article_social_view['platform'].eq(platform)
            ]['article_id'].nunique()
            if not article_social_view.empty else 0
        )
        coverage_rows.append({'Platform':platform.title(),'Coverage':100*linked/max(len(filtered_articles),1)})
    coverage_df=pd.DataFrame(coverage_rows)
    fig=px.bar(coverage_df, y='Platform', x='Coverage', orientation='h', text='Coverage',
               labels={'Coverage':'Website articles distributed (%)'})
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    fig.update_xaxes(range=[0,100])
    st.plotly_chart(fig, use_container_width=True, key='overview_platform_coverage')

    st.markdown('#### Publications over time')
    if not filtered_articles.empty:
        daily=(filtered_articles.dropna(subset=['published_at']).assign(Date=lambda d:d['published_at'].dt.date)
               .groupby(['Date','media']).size().reset_index(name='Articles'))
        fig=px.line(daily, x='Date', y='Articles', color='media', markers=True, labels={'media':'Partner'})
        st.plotly_chart(fig, use_container_width=True, key='overview_publications_over_time')

    st.markdown('#### Website views by publication date')
    if not perf_scope.empty:
        web_daily=(perf_scope.dropna(subset=['published_at']).assign(Date=lambda d:d['published_at'].dt.date)
                   .groupby(['Date','media'],as_index=False)['website_views'].sum())
        fig=px.line(web_daily, x='Date', y='website_views', color='media', markers=True,
                    labels={'website_views':'Website views','media':'Partner'})
        st.plotly_chart(fig, use_container_width=True, key='overview_website_views_over_time')

    st.markdown('#### Social reach by publication date')
    if not perf_scope.empty:
        reach_daily=(perf_scope.dropna(subset=['published_at']).assign(Date=lambda d:d['published_at'].dt.date)
                     .groupby(['Date','media'],as_index=False)['social_reach'].sum())
        fig=px.line(reach_daily, x='Date', y='social_reach', color='media', markers=True,
                    labels={'social_reach':'Social reach','media':'Partner'})
        st.plotly_chart(fig, use_container_width=True, key='overview_social_reach_over_time')

    st.markdown('#### Top-performing articles')
    if not perf_scope.empty:
        top=perf_scope.sort_values(['social_reach','website_views'],ascending=False).head(20)
        st.dataframe(top[[c for c in ['published_at','media','title','website_views','social_reach','social_posts','social_clicks','article_url'] if c in top.columns]],
                     hide_index=True,use_container_width=True,
                     column_config={'published_at':st.column_config.DatetimeColumn('Published',format='YYYY-MM-DD HH:mm'),
                                    'media':'Partner','title':'Article','article_url':st.column_config.LinkColumn('Open')})


# =========================================================
# 2. NETWORK PATTERNS
# =========================================================

with tabs[1]:
    st.subheader('Network Patterns')
    st.caption('Cross-partner references, shared stories and editorial relations.')

    cv_scope = citation_view_for(filtered_article_ids)
    flows = citation_flow_table(cv_scope)
    partners = sorted(filtered_articles['media'].dropna().unique())
    focus = st.selectbox('Reference network focus', ['All partners'] + partners, key='network_focus')

    st.markdown('#### Confirmed citation network')
    st.caption('Arrow direction follows information flow: source publication -> partner that cited it. The number on each arrow is the number of confirmed citations.')
    fig = directional_network_figure(flows, focus_partner=focus)
    if fig is None:
        st.info('No confirmed partner citations match the current filters.')
    else:
        st.plotly_chart(fig, use_container_width=True, key='network_confirmed_citation_network_v7_3')

    st.markdown('#### Confirmed references by partner')
    if not cv_scope.empty and {'citing_media','cited_media'}.issubset(cv_scope.columns):
        reference_rows=[]
        for p in partners:
            reference_rows.append({
                'Partner':p,
                'Cited by partners':int((cv_scope['cited_media'].eq(p) & cv_scope['citing_media'].ne(p)).sum()),
                'Cites partner publications':int((cv_scope['citing_media'].eq(p) & cv_scope['cited_media'].ne(p)).sum()),
            })
        ref_df=pd.DataFrame(reference_rows)
        ref_long=ref_df.melt(id_vars='Partner',var_name='Metric',value_name='Citations')
        fig=px.bar(ref_long,y='Partner',x='Citations',color='Metric',barmode='group',orientation='h')
        fig.update_layout(height=max(330,60*len(partners)+100))
        st.plotly_chart(fig, use_container_width=True, key='network_confirmed_references_by_partner')

    st.markdown('#### Stories across the network')
    if not story_memberships.empty:
        srows=[]
        scoped_members=story_memberships[story_memberships['article_id'].isin(filtered_article_ids)]
        for cid,g in scoped_members.groupby('cluster_id'):
            ids=set(g['article_id'].dropna())
            perf=associated_performance(ids)
            srows.append({'Story':g['cluster_title'].dropna().iloc[0] if g['cluster_title'].notna().any() else str(cid),
                          'Partners':g['media'].nunique(),'Articles':len(ids),'Website views':perf['website_views'],'Social reach':perf['social_reach']})
        spread=pd.DataFrame(srows).sort_values(['Partners','Social reach'],ascending=False)
        st.dataframe(spread,hide_index=True,use_container_width=True)

    st.markdown('#### Editorial relations')
    if rel_scope.empty:
        st.info('No possible pickups or follow-ups match the current filters.')
    else:
        id_to_media=article_meta.set_index('id')['media'].to_dict()
        rel=rel_scope.copy()
        rel['Source partner']=rel['source_article_id'].map(id_to_media)
        rel['Related partner']=rel['related_article_id'].map(id_to_media)
        rel=rel[rel['Source partner'].ne(rel['Related partner'])]
        counts=(rel.groupby(['Source partner','Related partner','relation_type']).size().reset_index(name='Relations'))
        counts['Pair']=counts['Source partner'].astype(str)+' -> '+counts['Related partner'].astype(str)
        fig=px.bar(counts,y='Pair',x='Relations',color='relation_type',orientation='h',
                   labels={'relation_type':'Relation type'})
        st.plotly_chart(fig, use_container_width=True, key='network_editorial_relations')
        st.dataframe(rel[[c for c in ['relation_type','Source partner','Related partner','confidence','evidence'] if c in rel.columns]].sort_values('confidence',ascending=False),
                     hide_index=True,use_container_width=True,
                     column_config={'confidence':st.column_config.NumberColumn('Relation confidence',format='%.2f')})

    st.markdown('#### Shared people, countries, organisations and places')
    if not entity_expanded.empty:
        overlap=[]
        e_scope=entity_expanded[entity_expanded['article_id'].isin(filtered_article_ids)]
        for (entity,etype),g in e_scope.groupby(['entity','entity_type']):
            ids=set(g['article_id'].dropna())
            if g['media'].nunique()<2:
                continue
            perf=associated_performance(ids)
            overlap.append({'Entity':entity,'Type':etype,'Partners':g['media'].nunique(),'Articles':len(ids),
                            'Website views':perf['website_views'],'Associated social reach':perf['social_reach']})
        overlap_df=pd.DataFrame(overlap).sort_values(['Partners','Associated social reach'],ascending=False)
        st.dataframe(overlap_df.head(100),hide_index=True,use_container_width=True)


# =========================================================
# 3. PARTNER PERSPECTIVE
# =========================================================

with tabs[2]:
    st.subheader('Partner Perspective')
    available = sorted(filtered_articles['media'].dropna().unique())
    if not available:
        st.info('No partners match the current filters.')
    else:
        chosen = st.selectbox('Partner', available, key='partner_perspective_selector')
        profile = partner_profile_dataframe(filtered_articles)

        st.markdown('#### Network contribution comparison')
        st.caption('Cell labels show actual values. Color shows each partner relative to the others for that metric. Lower response times are treated as stronger performance.')
        heat=contribution_heatmap(profile,chosen)
        if heat is not None:
            st.plotly_chart(heat, use_container_width=True, key='partner_perspective_network_contribution')

        row=profile[profile['Partner'].eq(chosen)].iloc[0]
        p1,p2,p3,p4,p5,p6=st.columns(6)
        p1.metric('Articles cited by partners',int(row['Articles cited by partners']))
        p2.metric('References to partner publications',int(row['References to partner publications']))
        p3.metric('Shared stories',int(row['Shared stories']))
        p4.metric('Median story response',format_hours(row['Median story response (h)']))
        p5.metric('Median citation response',format_hours(row['Median citation response (h)']))
        p6.metric('Social reach',compact_number(row['Social reach']))

        partner_articles=filtered_articles[filtered_articles['media'].eq(chosen)].copy()
        pids=set(partner_articles['id'])
        pp=article_performance[article_performance['article_id'].isin(pids)].copy()
        cv=citation_view_for(filtered_article_ids)

        st.markdown('#### Articles cited by partners')
        incoming=(cv[cv['cited_media'].eq(chosen) & cv['citing_media'].ne(chosen)].copy()
                  if not cv.empty else pd.DataFrame())
        if incoming.empty:
            st.info('No confirmed citations of this partner in the current filters.')
        else:
            by_citer=incoming.groupby('citing_media').size().reset_index(name='Citations').rename(columns={'citing_media':'Partner'})
            fig=px.bar(by_citer.sort_values('Citations'),y='Partner',x='Citations',orientation='h')
            st.plotly_chart(fig, use_container_width=True, key='partner_perspective_articles_cited')
            st.dataframe(incoming[[c for c in ['citing_published_at','citing_media','citing_title','cited_title','confidence','evidence_text'] if c in incoming.columns]],
                         hide_index=True,use_container_width=True)

        st.markdown('#### References to partner publications')
        outgoing=(cv[cv['citing_media'].eq(chosen) & cv['cited_media'].ne(chosen)].copy()
                  if not cv.empty else pd.DataFrame())
        if outgoing.empty:
            st.info('No confirmed references to other partners in the current filters.')
        else:
            by_source=outgoing.groupby('cited_media').size().reset_index(name='Citations').rename(columns={'cited_media':'Partner'})
            fig=px.bar(by_source.sort_values('Citations'),y='Partner',x='Citations',orientation='h')
            st.plotly_chart(fig, use_container_width=True, key='partner_perspective_references_to_partners')

        st.markdown('#### Distribution by platform')
        platform_rows=[]
        for platform in ['telegram','facebook','instagram']:
            linked=(article_social_view[article_social_view['article_id'].isin(pids) & article_social_view['platform'].eq(platform)]['article_id'].nunique()
                    if not article_social_view.empty else 0)
            platform_rows.append({'Platform':platform.title(),'Articles distributed':linked,'Coverage':100*linked/max(len(partner_articles),1)})
        platform_df=pd.DataFrame(platform_rows)
        fig=px.bar(platform_df,y='Platform',x='Coverage',orientation='h',text='Coverage',labels={'Coverage':'Website articles distributed (%)'})
        fig.update_traces(texttemplate='%{text:.0f}%',textposition='outside')
        fig.update_xaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True, key='partner_perspective_platform_distribution')

        st.markdown('#### Most referenced and distributed articles')
        if not pp.empty:
            incoming_map=(incoming.groupby('cited_article_id').size().to_dict() if not incoming.empty else {})
            rel_source=(article_relations[article_relations['source_article_id'].isin(pids)] if not article_relations.empty else pd.DataFrame())
            pickup_map=(rel_source[rel_source['relation_type'].eq('possible_pickup')].groupby('source_article_id').size().to_dict() if not rel_source.empty else {})
            follow_map=(rel_source[rel_source['relation_type'].eq('follow_up')].groupby('source_article_id').size().to_dict() if not rel_source.empty else {})
            table=pp.copy()
            table['Citations by partners']=table['article_id'].map(incoming_map).fillna(0).astype(int)
            table['Possible pickups']=table['article_id'].map(pickup_map).fillna(0).astype(int)
            table['Follow-ups']=table['article_id'].map(follow_map).fillna(0).astype(int)
            table=table.sort_values(['Citations by partners','social_reach','website_views'],ascending=False)
            st.dataframe(table[[c for c in ['published_at','title','website_views','social_reach','social_posts','Citations by partners','Possible pickups','Follow-ups','article_url'] if c in table.columns]].head(50),
                         hide_index=True,use_container_width=True,
                         column_config={'published_at':st.column_config.DatetimeColumn('Published',format='YYYY-MM-DD HH:mm'),'article_url':st.column_config.LinkColumn('Open')})


# =========================================================
# 4. CITATIONS
# =========================================================

with tabs[3]:
    st.subheader('Citations')
    cv=citation_view_for(filtered_article_ids)
    resolved=(cv.dropna(subset=['citing_media','cited_media']).copy()
              if not cv.empty and {'citing_media','cited_media'}.issubset(cv.columns) else pd.DataFrame())

    c1,c2,c3,c4=st.columns(4)
    c1.metric('Confirmed citations',len(resolved))
    c2.metric('Partner pairs',resolved[['citing_media','cited_media']].drop_duplicates().shape[0] if not resolved.empty else 0)
    c3.metric('Website views of citing articles',compact_number(sum_metric(resolved,'citing_web_views')) if not resolved.empty else '0')
    c4.metric('Social reach of citing articles',compact_number(sum_metric(resolved,'citing_social_reach')) if not resolved.empty else '0')

    st.markdown('#### Citation network')
    st.caption('Arrow direction is information flow: source publication -> partner that cited it. Numbers on arrows are citation counts.')
    flows=citation_flow_table(resolved)
    citation_partners=sorted(set(flows['Source'])|set(flows['Target'])) if not flows.empty else []
    focus=st.selectbox('Citation network focus',['All partners']+citation_partners,key='citation_focus')
    fig=directional_network_figure(flows,focus_partner=focus)
    if fig is None:
        st.info('No confirmed citations match the current filters.')
    else:
        st.plotly_chart(fig, use_container_width=True, key='citations_network_v7_3')

    st.markdown('#### Citation records')
    if resolved.empty:
        st.info('No citation records match the current filters.')
    else:
        table=resolved.sort_values('citing_published_at',ascending=False).reset_index(drop=True)
        display=table[[c for c in ['citing_published_at','cited_media','citing_media','cited_title','citing_title','citing_web_views','citing_social_reach','citing_topics','citing_people','citation_type','confidence'] if c in table.columns]].copy()
        event=st.dataframe(display,hide_index=True,use_container_width=True,on_select='rerun',selection_mode='single-row',key='citation_table_v7_3',
                           column_config={'citing_published_at':st.column_config.DatetimeColumn('Published',format='YYYY-MM-DD HH:mm'),
                                          'cited_media':'Source partner','citing_media':'Citing partner','cited_title':'Source article','citing_title':'Citing article',
                                          'confidence':st.column_config.NumberColumn('Confidence',format='%.2f')})
        selected=[]
        try:selected=list(event.selection.rows)
        except Exception:selected=[]
        if selected:
            r=table.iloc[selected[0]]
            st.markdown('#### Citation detail')
            st.markdown(f"**Source: {r.get('cited_media','---')}**")
            st.write(r.get('cited_title','---'))
            st.write(f"Website views: **{compact_number(r.get('cited_web_views',0))}**")
            st.write(f"Social reach: **{compact_number(r.get('cited_social_reach',0))}**")
            st.write('Topics: '+(r.get('cited_topics','') or '---'))
            st.write('People: '+(r.get('cited_people','') or '---'))
            st.write('Other entities: '+(r.get('cited_entities','') or '---'))
            st.markdown('**Information flow ->**')
            st.markdown(f"**Citing partner: {r.get('citing_media','---')}**")
            st.write(r.get('citing_title','---'))
            st.write(f"Website views: **{compact_number(r.get('citing_web_views',0))}**")
            st.write(f"Social reach: **{compact_number(r.get('citing_social_reach',0))}**")
            st.write('Topics: '+(r.get('citing_topics','') or '---'))
            st.write('People: '+(r.get('citing_people','') or '---'))
            st.write('Other entities: '+(r.get('citing_entities','') or '---'))
            if pd.notna(r.get('evidence_text')) and str(r.get('evidence_text')).strip():
                st.markdown('**Citation evidence**')
                st.write(r.get('evidence_text'))


# =========================================================
# 5. STORY JOURNEYS
# =========================================================

with tabs[4]:
    st.subheader('Story Journeys')
    if story_memberships.empty:
        st.info('No story clusters available.')
    else:
        story_scope=story_memberships[story_memberships['article_id'].isin(filtered_article_ids)].copy()
        if story_scope.empty:
            st.info('No stories match the current filters.')
        else:
            rows=[]
            for cid,g in story_scope.groupby('cluster_id'):
                g=g.sort_values('published_at')
                ids=set(g['article_id'].dropna())
                first=g.iloc[0]
                social=(social_links[social_links['article_id'].isin(ids)] if not social_links.empty else pd.DataFrame())
                topics_g=(topic_expanded[topic_expanded['article_id'].isin(ids)] if not topic_expanded.empty else pd.DataFrame())
                ents=(entity_expanded[entity_expanded['article_id'].isin(ids)] if not entity_expanded.empty else pd.DataFrame())
                perf=associated_performance(ids)
                cit=citation_view_for(ids)
                rel=(article_relations[article_relations['source_article_id'].isin(ids)|article_relations['related_article_id'].isin(ids)]
                     if not article_relations.empty else pd.DataFrame())
                first_published=first.get('first_published_at') if pd.notna(first.get('first_published_at')) else first.get('published_at')
                rows.append({
                    'cluster_id':cid,'First published':first_published,'First publisher':first.get('media'),'Story':first.get('cluster_title') or first.get('title'),
                    'Articles':len(ids),'Partners':g['media'].nunique(),'Social posts':social['social_post_id'].nunique() if not social.empty and 'social_post_id' in social.columns else len(social),
                    'Web views':perf['website_views'],'Social reach':perf['social_reach'],'Citations':len(cit.dropna(subset=['cited_article_id'])) if not cit.empty and 'cited_article_id' in cit.columns else 0,
                    'Possible pickups':int(rel['relation_type'].eq('possible_pickup').sum()) if not rel.empty else 0,
                    'Follow-ups':int(rel['relation_type'].eq('follow_up').sum()) if not rel.empty else 0,
                    'Cluster confidence':first.get('cluster_confidence'),
                    'Topics':join_unique(topics_g['topic'],limit=4) if not topics_g.empty else '',
                    'People':join_unique(ents[ents['entity_type'].eq('person')]['entity'],limit=4) if not ents.empty else '',
                    'Other entities':join_unique(ents[~ents['entity_type'].eq('person')]['entity'],limit=4) if not ents.empty else '',
                })
            story_index=pd.DataFrame(rows).sort_values('First published',ascending=False).reset_index(drop=True)
            st.caption('Select one row, then use Open selected story. The table is single-selection only.')
            display=story_index.drop(columns=['cluster_id'])
            event=st.dataframe(display,hide_index=True,use_container_width=True,on_select='rerun',selection_mode='single-row',key='story_table_v7_3',
                               column_config={'First published':st.column_config.DatetimeColumn('First published',format='YYYY-MM-DD HH:mm'),
                                              'Cluster confidence':st.column_config.NumberColumn('Cluster confidence',format='%.2f')})
            selected=[]
            try:selected=list(event.selection.rows)
            except Exception:selected=[]
            if selected:
                candidate_id=story_index.iloc[selected[0]]['cluster_id']
                if st.button('Open selected story ->',type='primary',key='open_story_v7'):
                    st.session_state['open_story_cluster_id']=candidate_id

            open_id=st.session_state.get('open_story_cluster_id')
            if open_id is not None and open_id in set(story_index['cluster_id']):
                selected_story=story_scope[story_scope['cluster_id'].eq(open_id)].copy().sort_values('published_at')
                first=selected_story.iloc[0]
                journey,social_story,article_ids=build_story_journey(selected_story)
                story_citations=citation_view_for(article_ids)
                if not story_citations.empty:
                    story_citations=story_citations[
                        story_citations['citing_article_id'].isin(article_ids)
                        & story_citations['cited_article_id'].isin(article_ids)
                    ]

                st.divider()
                st.markdown(f"### {first.get('cluster_title') or first.get('title')}")
                st.markdown('#### Story distribution map')
                st.caption('One timeline. Partner is encoded by color family; platform by shade. Node size reflects views/reach. Green arcs are confirmed citations; gray arrows show the chronological distribution path.')
                fig=story_distribution_figure(journey,story_citations)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key='story_distribution_map')

                perf=article_performance[article_performance['article_id'].isin(article_ids)]
                s1,s2,s3,s4,s5,s6=st.columns(6)
                s1.metric('Articles',len(article_ids))
                s2.metric('Partners',selected_story['media'].nunique())
                s3.metric('Social posts',journey['Type'].eq('Social post').sum() if not journey.empty else 0)
                s4.metric('Website views',compact_number(sum_metric(perf,'website_views')))
                s5.metric('Social reach',compact_number(sum_metric(perf,'social_reach')))
                s6.metric('Cluster confidence',f"{float(first.get('cluster_confidence')):.2f}" if pd.notna(first.get('cluster_confidence')) else '---')

                topics_g=(topic_expanded[topic_expanded['article_id'].isin(article_ids)] if not topic_expanded.empty else pd.DataFrame())
                ents=(entity_expanded[entity_expanded['article_id'].isin(article_ids)] if not entity_expanded.empty else pd.DataFrame())
                st.markdown('**Topics:** '+(join_unique(topics_g['topic'],limit=10) if not topics_g.empty else '---'))
                st.markdown('**People:** '+(join_unique(ents[ents['entity_type'].eq('person')]['entity'],limit=10) if not ents.empty else '---'))
                st.markdown('**Other entities:** '+(join_unique(ents[~ents['entity_type'].eq('person')]['entity'],limit=12) if not ents.empty else '---'))

                st.markdown('#### Publication chronology')
                if not journey.empty:
                    st.dataframe(journey[[c for c in ['Published','Partner','Channel','Content','Views','Reach','Clicks','URL'] if c in journey.columns]],
                                 hide_index=True,use_container_width=True,
                                 column_config={'Published':st.column_config.DatetimeColumn('Published',format='YYYY-MM-DD HH:mm'),'URL':st.column_config.LinkColumn('Open')})

                st.markdown('#### Distribution by partner and platform')
                dist=[]
                for partner in sorted(selected_story['media'].dropna().unique()):
                    pperf=perf[perf['media'].eq(partner)]
                    dist.append({'Partner':partner,'Channel':'Website views','Audience':sum_metric(pperf,'website_views')})
                    if not social_story.empty:
                        ps=social_story[social_story['media'].eq(partner)]
                        for platform,gp in ps.groupby('platform'):
                            reach=sum_metric(gp,'reach_count')
                            views=sum_metric(gp,'views_count')
                            dist.append({'Partner':partner,'Channel':str(platform).title()+' reach','Audience':reach if reach>0 else views})
                dist_df=pd.DataFrame(dist)
                if not dist_df.empty:
                    fig=px.bar(dist_df,y='Partner',x='Audience',color='Channel',barmode='group',orientation='h')
                    st.plotly_chart(fig, use_container_width=True, key='story_distribution_by_partner_platform')

                st.markdown('#### Evidence and editorial relations')
                relations=(article_relations[article_relations['source_article_id'].isin(article_ids)|article_relations['related_article_id'].isin(article_ids)]
                           if not article_relations.empty else pd.DataFrame())
                e1,e2,e3=st.columns(3)
                e1.metric('Confirmed citations',len(story_citations))
                e2.metric('Possible pickups',int(relations['relation_type'].eq('possible_pickup').sum()) if not relations.empty else 0)
                e3.metric('Follow-ups',int(relations['relation_type'].eq('follow_up').sum()) if not relations.empty else 0)
                if not relations.empty:
                    st.dataframe(relations[[c for c in ['source_article_id','related_article_id','relation_type','confidence','similarity_score','evidence'] if c in relations.columns]],
                                 hide_index=True,use_container_width=True,
                                 column_config={'confidence':st.column_config.NumberColumn('Relation confidence',format='%.2f')})

                st.markdown('#### Topics and entities by associated reach')
                theme_rows=[]
                if not topics_g.empty:
                    for topic,g in topics_g.groupby('topic'):
                        ids=set(g['article_id']); perf_t=associated_performance(ids)
                        theme_rows.append({'Theme':topic,'Type':'Topic','Articles':len(ids),'Partners':g['media'].nunique(),'Associated social reach':perf_t['social_reach']})
                if not ents.empty:
                    for (entity,etype),g in ents.groupby(['entity','entity_type']):
                        ids=set(g['article_id']); perf_t=associated_performance(ids)
                        theme_rows.append({'Theme':entity,'Type':etype.title(),'Articles':len(ids),'Partners':g['media'].nunique(),'Associated social reach':perf_t['social_reach']})
                tr=pd.DataFrame(theme_rows).sort_values('Associated social reach',ascending=False) if theme_rows else pd.DataFrame()
                if not tr.empty:
                    st.dataframe(tr.head(30),hide_index=True,use_container_width=True)


# =========================================================
# 6. PLATFORM DISTRIBUTION
# =========================================================

with tabs[5]:
    st.subheader('Platform Distribution')
    st.caption('Website articles connected to Telegram, Facebook, Instagram and other distribution channels.')

    social_scope=filtered_social_posts.copy()
    linked_scope=(article_social_view[article_social_view['article_id'].isin(filtered_article_ids)].copy()
                  if not article_social_view.empty else pd.DataFrame())
    p1,p2,p3,p4=st.columns(4)
    p1.metric('Social posts',len(social_scope))
    p2.metric('Linked social posts',linked_scope['social_post_id'].nunique() if not linked_scope.empty and 'social_post_id' in linked_scope.columns else len(linked_scope))
    p3.metric('Linked website articles',linked_scope['article_id'].nunique() if not linked_scope.empty else 0)
    p4.metric('Website to social coverage',f"{100*(linked_scope['article_id'].nunique() if not linked_scope.empty else 0)/max(len(filtered_articles),1):.0f}%")

    st.markdown('#### Platform output by partner')
    if not social_scope.empty and {'media','platform'}.issubset(social_scope.columns):
        output=social_scope.groupby(['media','platform']).size().reset_index(name='Posts')
        fig=px.bar(output,y='media',x='Posts',color='platform',orientation='h',barmode='stack',labels={'media':'Partner','platform':'Platform'})
        fig.update_layout(height=max(330,60*output['media'].nunique()+100))
        st.plotly_chart(fig, use_container_width=True, key='platform_output_by_partner')

    st.markdown('#### Website to platform coverage')
    coverage=[]
    for platform in ['telegram','facebook','instagram']:
        linked=(linked_scope[linked_scope['platform'].eq(platform)]['article_id'].nunique() if not linked_scope.empty else 0)
        coverage.append({'Platform':platform.title(),'Linked articles':linked,'Coverage':100*linked/max(len(filtered_articles),1)})
    cdf=pd.DataFrame(coverage)
    fig=px.bar(cdf,y='Platform',x='Coverage',orientation='h',text='Coverage',labels={'Coverage':'Website articles distributed (%)'})
    fig.update_traces(texttemplate='%{text:.0f}%',textposition='outside'); fig.update_xaxes(range=[0,100])
    st.plotly_chart(fig, use_container_width=True, key='platform_website_to_platform_coverage')

    st.markdown('#### Reach by platform')
    if not social_scope.empty and not social_latest.empty:
        sl=social_latest[social_latest['article_id'].isin(filtered_article_ids)].copy() if 'article_id' in social_latest.columns else pd.DataFrame()
        if not sl.empty:
            platform_perf=sl.groupby('platform',as_index=False).agg(Reach=('reach_count','sum'),Views=('views_count','sum'),Clicks=('clicks_count','sum'))
            long=platform_perf.melt(id_vars='platform',var_name='Metric',value_name='Value')
            fig=px.bar(long,y='platform',x='Value',color='Metric',barmode='group',orientation='h',labels={'platform':'Platform'})
            st.plotly_chart(fig, use_container_width=True, key='platform_reach_by_platform')

    st.markdown('#### Website to social links')
    if linked_scope.empty:
        st.info('No website to social links match the current filters.')
    else:
        st.dataframe(linked_scope[[c for c in ['article_published_at','media','title','article_url','platform','published_at','post_text','post_url','relation_method','relation_confidence'] if c in linked_scope.columns]].sort_values('article_published_at',ascending=False),
                     hide_index=True,use_container_width=True,
                     column_config={'article_published_at':st.column_config.DatetimeColumn('Website published',format='YYYY-MM-DD HH:mm'),
                                    'published_at':st.column_config.DatetimeColumn('Social published',format='YYYY-MM-DD HH:mm'),
                                    'article_url':st.column_config.LinkColumn('Article'),'post_url':st.column_config.LinkColumn('Post')})


# =========================================================
# 7. PUBLICATIONS EXPLORER
# =========================================================

with tabs[6]:
    st.subheader('Publications Explorer')
    scope_ids=set(filtered_article_ids)

    topic_options=sorted(topic_expanded[topic_expanded['article_id'].isin(scope_ids)]['topic'].dropna().unique()) if not topic_expanded.empty else []
    selected_topics=st.multiselect('Topics',topic_options,key='explorer_topics')
    if selected_topics:
        tids=set(topic_expanded[topic_expanded['topic'].isin(selected_topics)]['article_id'])
        scope_ids &= tids

    entity_options=sorted(entity_expanded[entity_expanded['article_id'].isin(scope_ids)]['entity'].dropna().unique()) if not entity_expanded.empty else []
    selected_entities=st.multiselect('People / entities',entity_options,key='explorer_entities')
    if selected_entities:
        eids=set(entity_expanded[entity_expanded['entity'].isin(selected_entities)]['article_id'])
        scope_ids &= eids

    keyword=st.text_input('Keyword in title or article text',key='explorer_keyword')
    results=filtered_articles[filtered_articles['id'].isin(scope_ids)].copy()
    if keyword.strip():
        kw=keyword.strip().lower()
        title=results['title'].fillna('').astype(str).str.lower()
        body=results['body_text'].fillna('').astype(str).str.lower() if 'body_text' in results.columns else pd.Series('',index=results.index)
        results=results[title.str.contains(kw,regex=False)|body.str.contains(kw,regex=False)]

    result_ids=set(results['id'])
    result_perf=article_performance[article_performance['article_id'].isin(result_ids)].copy()
    cv=citation_view_for(result_ids)
    cited_map=(cv[cv['cited_article_id'].isin(result_ids)].groupby('cited_article_id').size().to_dict() if not cv.empty and 'cited_article_id' in cv.columns else {})
    cluster_map=(story_memberships[story_memberships['article_id'].isin(result_ids)].groupby('article_id')['cluster_id'].nunique().to_dict() if not story_memberships.empty else {})

    if result_perf.empty:
        st.info('No publications match these filters.')
    else:
        result_perf['Topics']=result_perf['article_id'].map(article_topic_map).fillna('')
        result_perf['People']=result_perf['article_id'].map(article_people_map).fillna('')
        result_perf['Citations received']=result_perf['article_id'].map(cited_map).fillna(0).astype(int)
        result_perf['Stories']=result_perf['article_id'].map(cluster_map).fillna(0).astype(int)
        result_perf=result_perf.sort_values('published_at',ascending=False)
        st.dataframe(result_perf[[c for c in ['published_at','media','title','Topics','People','website_views','social_reach','social_posts','Citations received','Stories','article_url'] if c in result_perf.columns]],
                     hide_index=True,use_container_width=True,
                     column_config={'published_at':st.column_config.DatetimeColumn('Published',format='YYYY-MM-DD HH:mm'),'media':'Partner','title':'Article','article_url':st.column_config.LinkColumn('Open')})


# =========================================================
# 8. THEMES
# =========================================================

with tabs[7]:
    st.subheader('Themes')
    st.caption('Explore the network by topic, person, organisation, country, location or event.')

    theme_type=st.selectbox('Explore by',['Topic','Person','Organization','Country','Location','Event'],key='theme_type')
    stats=[]
    item_to_ids={}

    if theme_type=='Topic':
        source=topic_expanded[topic_expanded['article_id'].isin(filtered_article_ids)] if not topic_expanded.empty else pd.DataFrame()
        if not source.empty:
            for item,g in source.groupby('topic'):
                ids=set(g['article_id']); item_to_ids[item]=ids
                row=theme_stats(ids,item,'Topic')
                if row:stats.append(row)
    else:
        type_map={'Person':'person','Organization':'organization','Country':'country','Location':'location','Event':'event'}
        etype=type_map[theme_type]
        source=(entity_expanded[entity_expanded['article_id'].isin(filtered_article_ids) & entity_expanded['entity_type'].eq(etype)]
                if not entity_expanded.empty else pd.DataFrame())
        if not source.empty:
            for item,g in source.groupby('entity'):
                ids=set(g['article_id']); item_to_ids[item]=ids
                row=theme_stats(ids,item,theme_type)
                if row:stats.append(row)

    stats_df=pd.DataFrame(stats)
    if stats_df.empty:
        st.info('No theme data matches the current filters.')
    else:
        stats_df=stats_df.sort_values(['Partners','Social reach','Articles'],ascending=False)
        st.markdown('#### Network spread')
        st.dataframe(stats_df,hide_index=True,use_container_width=True)

        st.markdown('#### Reach by theme')
        chart=stats_df.sort_values('Social reach',ascending=False).head(20).sort_values('Social reach')
        fig=px.bar(chart,y='Theme',x='Social reach',orientation='h',hover_data=['Articles','Partners','Shared stories','Citations received'])
        st.plotly_chart(fig, use_container_width=True, key='themes_reach_by_theme')

        selected_theme=st.selectbox('Open theme',stats_df['Theme'].tolist(),key='theme_selector')
        ids=item_to_ids.get(selected_theme,set())
        subset=filtered_articles[filtered_articles['id'].isin(ids)].copy()
        perf=article_performance[article_performance['article_id'].isin(ids)].copy()
        selected_stats=stats_df[stats_df['Theme'].eq(selected_theme)].iloc[0]

        t1,t2,t3,t4,t5,t6=st.columns(6)
        t1.metric('Articles',int(selected_stats['Articles']))
        t2.metric('Partners',int(selected_stats['Partners']))
        t3.metric('Shared stories',int(selected_stats['Shared stories']))
        t4.metric('Citations received',int(selected_stats['Citations received']))
        t5.metric('Website views',compact_number(selected_stats['Website views']))
        t6.metric('Social reach',compact_number(selected_stats['Social reach']))

        st.markdown('#### Distribution across partners')
        if not perf.empty:
            partner_perf=perf.groupby('media',as_index=False).agg(Articles=('article_id','nunique'),Website_views=('website_views','sum'),Social_reach=('social_reach','sum'))
            long=partner_perf.melt(id_vars='media',value_vars=['Website_views','Social_reach'],var_name='Metric',value_name='Audience')
            long['Metric']=long['Metric'].str.replace('_',' ',regex=False)
            fig=px.bar(long,y='media',x='Audience',color='Metric',orientation='h',barmode='group',labels={'media':'Partner'})
            st.plotly_chart(fig, use_container_width=True, key='themes_distribution_across_partners')

        st.markdown('#### Activity over time')
        if not subset.empty:
            daily=(subset.dropna(subset=['published_at']).assign(Date=lambda d:d['published_at'].dt.date)
                   .groupby(['Date','media']).size().reset_index(name='Articles'))
            fig=px.line(daily,x='Date',y='Articles',color='media',markers=True,labels={'media':'Partner'})
            st.plotly_chart(fig, use_container_width=True, key='themes_activity_over_time')

        st.markdown('#### Publications')
        article_table(subset,limit=100)

        st.markdown('#### Associated topics and entities')
        associated=[]
        if theme_type!='Topic' and not topic_expanded.empty:
            tg=topic_expanded[topic_expanded['article_id'].isin(ids)]
            for item,g in tg.groupby('topic'):
                associated.append({'Associated item':item,'Type':'Topic','Articles':g['article_id'].nunique()})
        if not entity_expanded.empty:
            eg=entity_expanded[entity_expanded['article_id'].isin(ids)]
            for (item,etype),g in eg.groupby(['entity','entity_type']):
                if item==selected_theme:continue
                associated.append({'Associated item':item,'Type':etype.title(),'Articles':g['article_id'].nunique()})
        assoc=pd.DataFrame(associated).sort_values(['Articles','Associated item'],ascending=[False,True]) if associated else pd.DataFrame()
        if not assoc.empty:
            st.dataframe(assoc.head(50),hide_index=True,use_container_width=True)


st.caption(
    'Demo: Media Partners shared intelligence layer. All August 2026 content and performance figures in this environment are synthetic. Confirmed citations, same-story clusters, possible pickups and follow-ups are shown as separate evidence types.'
)
