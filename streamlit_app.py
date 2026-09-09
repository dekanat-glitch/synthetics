
import os
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
# Tabs
# ---------------------------------------------------------

tabs = st.tabs(
    [
        "Overview",
        "Topics",
        "People & Entities",
        "Publications Explorer",
        "Citations",
        "Story Journeys",
        "Social Distribution",
        "Performance",
    ]
)

# =========================================================
# 1. OVERVIEW
# =========================================================

with tabs[0]:

    st.subheader("Network overview")

    filtered_topics = (
        topic_expanded[
            topic_expanded["article_id"].isin(filtered_article_ids)
        ]
        if not topic_expanded.empty
        else pd.DataFrame()
    )

    filtered_entities = (
        entity_expanded[
            entity_expanded["article_id"].isin(filtered_article_ids)
        ]
        if not entity_expanded.empty
        else pd.DataFrame()
    )

    filtered_citations = (
        citations[
            citations["citing_article_id"].isin(filtered_article_ids)
        ]
        if (
            not citations.empty
            and "citing_article_id" in citations.columns
        )
        else pd.DataFrame()
    )

    filtered_story_memberships = (
        story_memberships[
            story_memberships["article_id"].isin(filtered_article_ids)
        ]
        if not story_memberships.empty
        else pd.DataFrame()
    )

    linked_article_ids = set()

    if not social_links.empty:
        linked_article_ids = (
            set(social_links["article_id"])
            & filtered_article_ids
        )

    possible_pickups = 0

    if (
        not article_relations.empty
        and "relation_type" in article_relations.columns
    ):
        rel = article_relations[
            article_relations["relation_type"] == "possible_pickup"
        ]

        if "source_article_id" in rel.columns:
            rel = rel[
                rel["source_article_id"].isin(filtered_article_ids)
            ]

        possible_pickups = len(rel)

    telegram_links_count = 0

    if not article_social_view.empty:
        telegram_view = article_social_view[
            article_social_view["platform"].eq("telegram")
            & article_social_view["article_id"].isin(filtered_article_ids)
        ]
        telegram_links_count = len(telegram_view)

    overview_perf = article_performance[
        article_performance["article_id"].isin(filtered_article_ids)
    ].copy()

    website_views = sum_metric(overview_perf, "website_views")
    social_views = sum_metric(overview_perf, "social_views")
    social_reach = sum_metric(overview_perf, "social_reach")
    social_interaction_count = sum_metric(
        overview_perf,
        "social_interactions"
    )

    shared_story_count = (
        filtered_story_memberships["cluster_id"].nunique()
        if not filtered_story_memberships.empty
        else 0
    )

    coverage_pct = (
        100 * len(linked_article_ids) / len(filtered_articles)
        if len(filtered_articles)
        else 0
    )

    # The first row contains the numbers partners normally ask for first.
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric(
        "Website articles",
        metric_number(len(filtered_articles))
    )

    k2.metric(
        "Social posts",
        metric_number(len(filtered_social_posts))
    )

    k3.metric(
        "Website views",
        compact_number(website_views)
    )

    k4.metric(
        "Social reach",
        compact_number(social_reach)
    )

    k5.metric(
        "Direct citations",
        metric_number(len(filtered_citations))
    )

    k6.metric(
        "Shared stories",
        metric_number(shared_story_count)
    )

    q1, q2, q3, q4, q5, q6 = st.columns(6)

    q1.metric(
        "Website → social coverage",
        f"{coverage_pct:.0f}%"
    )

    q2.metric(
        "Articles linked to social",
        metric_number(len(linked_article_ids))
    )

    q3.metric(
        "Website ↔ Telegram",
        metric_number(telegram_links_count)
    )

    q4.metric(
        "Social views",
        compact_number(social_views)
    )

    q5.metric(
        "Social interactions",
        compact_number(social_interaction_count)
    )

    q6.metric(
        "Possible pickups",
        metric_number(possible_pickups)
    )

    st.caption(
        "Social reach is the sum of the latest platform reach snapshots; "
        "it is not a deduplicated unique-audience figure."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Publications by partner")

        by_media = (
            filtered_articles
            .groupby("media", dropna=False)
            .size()
            .reset_index(name="articles")
            .sort_values("articles", ascending=False)
        )

        if len(by_media):
            fig = px.bar(
                by_media,
                x="media",
                y="articles",
                labels={
                    "media": "Partner",
                    "articles": "Articles"
                }
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("#### Audience by partner")

        partner_perf = (
            overview_perf
            .groupby("media", as_index=False)
            .agg(
                website_views=("website_views", "sum"),
                social_reach=("social_reach", "sum"),
            )
        )

        if not partner_perf.empty:
            partner_long = partner_perf.melt(
                id_vars=["media"],
                value_vars=["website_views", "social_reach"],
                var_name="metric",
                value_name="value"
            )
            partner_long["metric"] = partner_long["metric"].map({
                "website_views": "Website views",
                "social_reach": "Social reach",
            })

            fig = px.bar(
                partner_long,
                x="media",
                y="value",
                color="metric",
                barmode="group",
                labels={
                    "media": "Partner",
                    "value": "Audience metric",
                    "metric": "Metric",
                }
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:

        st.markdown("#### Publishing over time")

        trend = (
            filtered_articles
            .dropna(subset=["published_at"])
            .assign(day=lambda x: x["published_at"].dt.date)
            .groupby(["day", "media"])
            .size()
            .reset_index(name="articles")
        )

        if len(trend):
            fig = px.line(
                trend,
                x="day",
                y="articles",
                color="media",
                markers=True,
                labels={
                    "day": "Date",
                    "articles": "Articles",
                    "media": "Partner"
                }
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("#### Social reach by platform")

        social_scope = social_latest.copy()

        if not social_scope.empty:
            if selected_media and "media" in social_scope.columns:
                social_scope = social_scope[
                    social_scope["media"].isin(selected_media)
                ]

            if (
                selected_dates
                and isinstance(selected_dates, (tuple, list))
                and len(selected_dates) == 2
                and "post_published_at" in social_scope.columns
            ):
                start_date, end_date = selected_dates
                social_scope = social_scope[
                    social_scope["post_published_at"]
                    .dt.date
                    .between(start_date, end_date)
                ]

        if not social_scope.empty:
            by_platform = (
                social_scope
                .groupby("platform", as_index=False)
                .agg(reach=("reach_count", "sum"))
                .sort_values("reach", ascending=False)
            )

            fig = px.bar(
                by_platform,
                x="platform",
                y="reach",
                labels={
                    "platform": "Platform",
                    "reach": "Reach"
                }
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No social metrics match these filters.")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Top topics")

        if not filtered_topics.empty:
            top_topics = (
                filtered_topics
                .groupby("topic")["article_id"]
                .nunique()
                .reset_index(name="articles")
                .sort_values("articles", ascending=False)
                .head(12)
            )

            fig = px.bar(
                top_topics,
                x="articles",
                y="topic",
                orientation="h",
                labels={
                    "articles": "Articles",
                    "topic": "Topic"
                }
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Topic data is not available.")

    with right:

        st.markdown("#### Top people")

        people = (
            filtered_entities[
                filtered_entities["entity_type"] == "person"
            ]
            if (
                not filtered_entities.empty
                and "entity_type" in filtered_entities.columns
            )
            else pd.DataFrame()
        )

        if not people.empty:
            top_people = (
                people
                .groupby("entity")["article_id"]
                .nunique()
                .reset_index(name="articles")
                .sort_values("articles", ascending=False)
                .head(12)
            )

            fig = px.bar(
                top_people,
                x="articles",
                y="entity",
                orientation="h",
                labels={
                    "articles": "Articles",
                    "entity": "Person"
                }
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("People data is not available.")


# =========================================================
# 2. TOPICS
# =========================================================

with tabs[1]:

    st.subheader("Topic coverage")

    if topic_expanded.empty:

        st.info("No topic assignments available.")

    else:

        topic_options = sorted(
            topic_expanded[
                "topic"
            ].dropna().unique()
        )

        selected_topic = st.selectbox(
            "Topic",
            topic_options
        )

        role_options = [
            "All roles",
            "primary",
            "secondary"
        ]

        selected_role = st.radio(
            "Topic role",
            role_options,
            horizontal=True
        )

        subset = topic_expanded[
            (
                topic_expanded[
                    "topic"
                ] == selected_topic
            )
            &
            (
                topic_expanded[
                    "article_id"
                ].isin(
                    filtered_article_ids
                )
            )
        ].copy()

        if (
            selected_role != "All roles"
            and "role" in subset.columns
        ):
            subset = subset[
                subset[
                    "role"
                ] == selected_role
            ]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Publications",
            subset[
                "article_id"
            ].nunique()
        )

        m2.metric(
            "Partners",
            subset[
                "media"
            ].nunique()
        )

        m3.metric(
            "Primary assignments",
            int(
                (
                    subset[
                        "role"
                    ] == "primary"
                ).sum()
            )
            if "role" in subset.columns
            else "—"
        )

        left, right = st.columns(2)

        with left:

            st.markdown(
                f"#### Who publishes about {selected_topic}"
            )

            by_media = (
                subset
                .groupby("media")[
                    "article_id"
                ]
                .nunique()
                .reset_index(
                    name="articles"
                )
                .sort_values(
                    "articles",
                    ascending=False
                )
            )

            if len(by_media):

                fig = px.bar(
                    by_media,
                    x="media",
                    y="articles",
                    labels={
                        "media": "Partner",
                        "articles": "Articles"
                    }
                )

                fig.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with right:

            st.markdown(
                "#### Topic publishing over time"
            )

            trend = (
                subset
                .dropna(
                    subset=[
                        "published_at"
                    ]
                )
                .assign(
                    day=lambda x:
                    x[
                        "published_at"
                    ].dt.date
                )
                .groupby(
                    [
                        "day",
                        "media"
                    ]
                )[
                    "article_id"
                ]
                .nunique()
                .reset_index(
                    name="articles"
                )
            )

            if len(trend):

                fig = px.line(
                    trend,
                    x="day",
                    y="articles",
                    color="media",
                    markers=True,
                    labels={
                        "day": "Date",
                        "articles": "Articles",
                        "media": "Partner"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("#### Recent publications")

        topic_articles = (
            subset[
                [
                    "article_id",
                    "published_at",
                    "media",
                    "title",
                    "article_url"
                ]
            ]
            .drop_duplicates(
                "article_id"
            )
        )

        article_table(
            topic_articles,
            limit=300
        )


# =========================================================
# 3. PEOPLE & ENTITIES
# =========================================================

with tabs[2]:

    st.subheader("People & entities")

    if entity_expanded.empty:

        st.info("No entity assignments available.")

    else:

        available_types = sorted(
            entity_expanded[
                "entity_type"
            ].dropna().unique()
        )

        default_type_index = (
            available_types.index(
                "person"
            )
            if "person" in available_types
            else 0
        )

        selected_type = st.selectbox(
            "Entity type",
            available_types,
            index=default_type_index
        )

        type_subset = entity_expanded[
            (
                entity_expanded[
                    "entity_type"
                ] == selected_type
            )
            &
            (
                entity_expanded[
                    "article_id"
                ].isin(
                    filtered_article_ids
                )
            )
        ].copy()

        search = st.text_input(
            "Search",
            placeholder=(
                "e.g. Zelenskyy, EU, Kyiv, CIA..."
            )
        ).strip()

        entity_options = sorted(
            type_subset[
                "entity"
            ].dropna().unique()
        )

        if search:
            entity_options = [
                x
                for x in entity_options
                if search.lower()
                in str(x).lower()
            ]

        left, right = st.columns(
            [1, 2]
        )

        with left:

            st.markdown("#### Most mentioned")

            ranking = (
                type_subset
                .groupby("entity")
                .agg(
                    articles=(
                        "article_id",
                        "nunique"
                    ),
                    partners=(
                        "media",
                        "nunique"
                    )
                )
                .reset_index()
                .sort_values(
                    [
                        "articles",
                        "partners"
                    ],
                    ascending=[
                        False,
                        False
                    ]
                )
                .head(30)
            )

            st.dataframe(
                ranking,
                hide_index=True,
                use_container_width=True
            )

        with right:

            if entity_options:

                selected_entity = st.selectbox(
                    "Inspect entity",
                    entity_options
                )

                selected_rows = type_subset[
                    type_subset[
                        "entity"
                    ] == selected_entity
                ].copy()

                c1, c2 = st.columns(2)

                c1.metric(
                    "Publications",
                    selected_rows[
                        "article_id"
                    ].nunique()
                )

                c2.metric(
                    "Partners",
                    selected_rows[
                        "media"
                    ].nunique()
                )

                by_media = (
                    selected_rows
                    .groupby("media")[
                        "article_id"
                    ]
                    .nunique()
                    .reset_index(
                        name="articles"
                    )
                )

                if len(by_media):

                    fig = px.bar(
                        by_media,
                        x="media",
                        y="articles",
                        labels={
                            "media": "Partner",
                            "articles": "Articles"
                        }
                    )

                    fig.update_layout(
                        showlegend=False
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.markdown(
                    f"#### Recent publications mentioning {selected_entity}"
                )

                selected_articles = (
                    selected_rows[
                        [
                            "article_id",
                            "published_at",
                            "media",
                            "title",
                            "article_url"
                        ]
                    ]
                    .drop_duplicates(
                        "article_id"
                    )
                )

                article_table(
                    selected_articles
                )

                # Co-occurring entities
                selected_article_ids = set(
                    selected_rows[
                        "article_id"
                    ]
                )

                co = entity_expanded[
                    (
                        entity_expanded[
                            "article_id"
                        ].isin(
                            selected_article_ids
                        )
                    )
                    &
                    (
                        entity_expanded[
                            "entity"
                        ] != selected_entity
                    )
                ]

                if not co.empty:

                    co_rank = (
                        co
                        .groupby(
                            [
                                "entity",
                                "entity_type"
                            ]
                        )[
                            "article_id"
                        ]
                        .nunique()
                        .reset_index(
                            name="shared_articles"
                        )
                        .sort_values(
                            "shared_articles",
                            ascending=False
                        )
                        .head(20)
                    )

                    st.markdown(
                        "#### Often appears in the same publications"
                    )

                    st.dataframe(
                        co_rank,
                        hide_index=True,
                        use_container_width=True
                    )

            else:
                st.info(
                    "No entities match this search and filter."
                )


# =========================================================
# 4. PUBLICATIONS EXPLORER
# =========================================================

with tabs[3]:

    st.subheader("Publications explorer")

    explorer = filtered_articles.copy()

    col1, col2, col3 = st.columns(3)

    with col1:

        topic_filter = st.multiselect(
            "Topics",
            sorted(
                topic_expanded[
                    "topic"
                ].dropna().unique()
            )
            if not topic_expanded.empty
            else []
        )

    with col2:

        entity_type_filter = st.selectbox(
            "Entity type",
            [
                "Any"
            ]
            + (
                sorted(
                    entity_expanded[
                        "entity_type"
                    ].dropna().unique()
                )
                if not entity_expanded.empty
                else []
            ),
            key="explorer_entity_type"
        )

    with col3:

        keyword = st.text_input(
            "Keyword",
            placeholder="Search title or article text"
        ).strip()

    entity_options = []

    if not entity_expanded.empty:

        entity_base = entity_expanded

        if entity_type_filter != "Any":
            entity_base = entity_base[
                entity_base[
                    "entity_type"
                ] == entity_type_filter
            ]

        entity_options = sorted(
            entity_base[
                "entity"
            ].dropna().unique()
        )

    entity_filter = st.multiselect(
        "Entities / people",
        entity_options
    )

    if topic_filter:

        ids = set(
            topic_expanded[
                topic_expanded[
                    "topic"
                ].isin(
                    topic_filter
                )
            ][
                "article_id"
            ]
        )

        explorer = explorer[
            explorer[
                "id"
            ].isin(ids)
        ]

    if entity_filter:

        ids = set(
            entity_expanded[
                entity_expanded[
                    "entity"
                ].isin(
                    entity_filter
                )
            ][
                "article_id"
            ]
        )

        explorer = explorer[
            explorer[
                "id"
            ].isin(ids)
        ]

    if keyword:

        title_match = (
            explorer[
                "title"
            ]
            .fillna("")
            .str.contains(
                keyword,
                case=False,
                regex=False
            )
        )

        if "body_text" in explorer.columns:
            body_match = (
                explorer[
                    "body_text"
                ]
                .fillna("")
                .str.contains(
                    keyword,
                    case=False,
                    regex=False
                )
            )
        else:
            body_match = False

        explorer = explorer[
            title_match
            | body_match
        ]

    st.metric(
        "Matching publications",
        len(explorer)
    )

    article_table(
        explorer,
        limit=500
    )

    if not article_social_view.empty:

        explorer_links = article_social_view[
            article_social_view[
                "article_id"
            ].isin(
                set(
                    explorer["id"]
                )
            )
        ].copy()

        telegram_links = explorer_links[
            explorer_links[
                "platform"
            ].eq("telegram")
        ].copy()

        if not telegram_links.empty:

            st.markdown(
                "#### Website ↔ Telegram links in these results"
            )

            show_cols = [
                c for c in [
                    "article_published_at",
                    "media",
                    "title",
                    "article_url",
                    "published_at",
                    "post_text",
                    "post_url",
                    "relation_method",
                    "relation_confidence",
                ]
                if c in telegram_links.columns
            ]

            st.dataframe(
                telegram_links[
                    show_cols
                ].sort_values(
                    "article_published_at",
                    ascending=False
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "article_published_at":
                        st.column_config.DatetimeColumn(
                            "Website published",
                            format="YYYY-MM-DD HH:mm"
                        ),
                    "media": "Partner",
                    "title": "Website article",
                    "article_url":
                        st.column_config.LinkColumn(
                            "Open website"
                        ),
                    "published_at":
                        st.column_config.DatetimeColumn(
                            "Telegram published",
                            format="YYYY-MM-DD HH:mm"
                        ),
                    "post_text": "Telegram post",
                    "post_url":
                        st.column_config.LinkColumn(
                            "Open Telegram"
                        ),
                    "relation_method": "Link method",
                    "relation_confidence":
                        st.column_config.NumberColumn(
                            "Confidence",
                            format="%.2f"
                        ),
                },
            )


# =========================================================
# 5. CITATIONS
# =========================================================

with tabs[4]:

    st.subheader("Citations")

    if citations.empty:
        st.info("No citation records available.")

    else:
        citation_view = citations.copy()

        if "citing_article_id" in citation_view.columns:
            citation_view = citation_view[
                citation_view["citing_article_id"].isin(
                    filtered_article_ids
                )
            ]

        # Add topics, entities and performance to both sides of a citation.
        for prefix, id_col in [
            ("citing", "citing_article_id"),
            ("cited", "cited_article_id"),
        ]:
            if id_col in citation_view.columns:
                citation_view[f"{prefix}_topics"] = (
                    citation_view[id_col].map(article_topic_map).fillna("")
                )
                citation_view[f"{prefix}_people"] = (
                    citation_view[id_col].map(article_people_map).fillna("")
                )
                citation_view[f"{prefix}_entities"] = (
                    citation_view[id_col]
                    .map(article_other_entities_map)
                    .fillna("")
                )

                citation_view[f"{prefix}_web_views"] = (
                    citation_view[id_col]
                    .map(
                        lambda x: article_metrics(x).get(
                            "website_views", 0
                        )
                    )
                )
                citation_view[f"{prefix}_social_reach"] = (
                    citation_view[id_col]
                    .map(
                        lambda x: article_metrics(x).get(
                            "social_reach", 0
                        )
                    )
                )

        direct_partner = (
            citation_view[citation_view["cited_article_id"].notna()]
            if "cited_article_id" in citation_view.columns
            else pd.DataFrame()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Citation records", len(citation_view))
        c2.metric("Resolved partner citations", len(direct_partner))
        c3.metric(
            "Partner pairs",
            (
                direct_partner[
                    ["citing_media", "cited_media"]
                ]
                .dropna()
                .drop_duplicates()
                .shape[0]
            )
            if (
                not direct_partner.empty
                and {"citing_media", "cited_media"}.issubset(
                    direct_partner.columns
                )
            )
            else 0
        )
        c4.metric(
            "Reach of citing publications",
            compact_number(
                sum_metric(citation_view, "citing_social_reach")
            )
        )

        if (
            not direct_partner.empty
            and {"citing_media", "cited_media"}.issubset(
                direct_partner.columns
            )
        ):
            left, right = st.columns(2)

            with left:
                st.markdown("#### Partner-to-partner citation flow")

                flows = (
                    direct_partner
                    .dropna(subset=["citing_media", "cited_media"])
                    .groupby(["citing_media", "cited_media"])
                    .size()
                    .reset_index(name="citations")
                )

                if len(flows):
                    labels = sorted(
                        set(flows["citing_media"])
                        | set(flows["cited_media"])
                    )
                    label_index = {
                        label: i
                        for i, label in enumerate(labels)
                    }

                    fig = go.Figure(
                        data=[
                            go.Sankey(
                                node=dict(label=labels),
                                link=dict(
                                    source=[
                                        label_index[x]
                                        for x in flows["citing_media"]
                                    ],
                                    target=[
                                        label_index[x]
                                        for x in flows["cited_media"]
                                    ],
                                    value=flows["citations"].tolist()
                                )
                            )
                        ]
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown("#### Distribution of citing publications")

                impact = direct_partner.copy()
                impact["pair"] = (
                    impact["citing_media"].astype(str)
                    + " → "
                    + impact["cited_media"].astype(str)
                )
                impact = (
                    impact
                    .groupby("pair", as_index=False)
                    .agg(
                        citations=("id", "count"),
                        social_reach=("citing_social_reach", "sum"),
                        website_views=("citing_web_views", "sum"),
                    )
                    .sort_values("social_reach", ascending=False)
                )

                impact_long = impact.melt(
                    id_vars=["pair"],
                    value_vars=["website_views", "social_reach"],
                    var_name="metric",
                    value_name="value"
                )
                impact_long["metric"] = impact_long["metric"].map({
                    "website_views": "Website views",
                    "social_reach": "Social reach",
                })

                fig = px.bar(
                    impact_long,
                    x="pair",
                    y="value",
                    color="metric",
                    barmode="group",
                    labels={
                        "pair": "Citation direction",
                        "value": "Audience metric",
                        "metric": "Metric",
                    }
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Citation records")
        st.caption(
            "Click a citation row to inspect topics, people, entities, "
            "evidence and performance on both sides."
        )

        citation_table = citation_view.sort_values(
            "citing_published_at",
            ascending=False
        ).reset_index(drop=True)

        citation_display = citation_table[
            [
                c for c in [
                    "citing_published_at",
                    "citing_media",
                    "citing_title",
                    "cited_media",
                    "cited_title",
                    "citing_web_views",
                    "citing_social_reach",
                    "citing_topics",
                    "citing_people",
                    "citation_type",
                    "confidence",
                ]
                if c in citation_table.columns
            ]
        ].copy()

        citation_event = st.dataframe(
            citation_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="citation_records_table",
            column_config={
                "citing_published_at": st.column_config.DatetimeColumn(
                    "Published",
                    format="YYYY-MM-DD HH:mm"
                ),
                "citing_media": "Citing partner",
                "citing_title": "Citing article",
                "cited_media": "Cited partner",
                "cited_title": "Cited article",
                "citing_web_views": st.column_config.NumberColumn(
                    "Citing web views",
                    format="%d"
                ),
                "citing_social_reach": st.column_config.NumberColumn(
                    "Citing social reach",
                    format="%d"
                ),
                "citing_topics": "Topics",
                "citing_people": "People",
                "citation_type": "Citation type",
                "confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f"
                ),
            }
        )

        selected_citation_rows = []
        try:
            selected_citation_rows = list(
                citation_event.selection.rows
            )
        except Exception:
            selected_citation_rows = []

        if selected_citation_rows:
            citation_row = citation_table.iloc[
                selected_citation_rows[0]
            ]

            st.markdown("#### Citation detail")

            left, right = st.columns(2)

            with left:
                st.markdown(
                    f"**Citing: {citation_row.get('citing_media', '—')}**"
                )
                st.write(citation_row.get("citing_title", "—"))
                st.write(
                    f"Website views: **{compact_number(citation_row.get('citing_web_views', 0))}**"
                )
                st.write(
                    f"Social reach: **{compact_number(citation_row.get('citing_social_reach', 0))}**"
                )
                st.write(
                    "Topics: "
                    + (citation_row.get("citing_topics", "") or "—")
                )
                st.write(
                    "People: "
                    + (citation_row.get("citing_people", "") or "—")
                )
                st.write(
                    "Other entities: "
                    + (citation_row.get("citing_entities", "") or "—")
                )

                if pd.notna(citation_row.get("citing_url")):
                    st.link_button(
                        "Open citing article",
                        citation_row.get("citing_url")
                    )

            with right:
                st.markdown(
                    f"**Cited: {citation_row.get('cited_media', '—')}**"
                )
                st.write(citation_row.get("cited_title", "—"))
                st.write(
                    f"Website views: **{compact_number(citation_row.get('cited_web_views', 0))}**"
                )
                st.write(
                    f"Social reach: **{compact_number(citation_row.get('cited_social_reach', 0))}**"
                )
                st.write(
                    "Topics: "
                    + (citation_row.get("cited_topics", "") or "—")
                )
                st.write(
                    "People: "
                    + (citation_row.get("cited_people", "") or "—")
                )
                st.write(
                    "Other entities: "
                    + (citation_row.get("cited_entities", "") or "—")
                )

                if pd.notna(citation_row.get("cited_url")):
                    st.link_button(
                        "Open cited article",
                        citation_row.get("cited_url")
                    )

            evidence = citation_row.get("evidence_text")
            if pd.notna(evidence) and str(evidence).strip():
                st.markdown("**Citation evidence**")
                st.write(evidence)


# =========================================================
# 6. STORY JOURNEYS
# =========================================================

with tabs[5]:

    st.subheader("Story journeys")

    if story_memberships.empty:
        st.info("No confirmed story clusters available.")

    else:
        story_scope = story_memberships[
            story_memberships["article_id"].isin(filtered_article_ids)
        ].copy()

        if story_scope.empty:
            st.info("No story clusters match the current filters.")

        else:
            story_rows = []

            for cluster_id, cluster_df in story_scope.groupby("cluster_id"):
                cluster_df = cluster_df.sort_values("published_at")
                first_row = cluster_df.iloc[0]
                article_ids = set(cluster_df["article_id"].dropna())

                story_social = (
                    social_links[
                        social_links["article_id"].isin(article_ids)
                    ]
                    if not social_links.empty
                    else pd.DataFrame()
                )

                story_topics = (
                    topic_expanded[
                        topic_expanded["article_id"].isin(article_ids)
                    ]
                    if not topic_expanded.empty
                    else pd.DataFrame()
                )

                story_entities = (
                    entity_expanded[
                        entity_expanded["article_id"].isin(article_ids)
                    ]
                    if not entity_expanded.empty
                    else pd.DataFrame()
                )

                people_text = ""
                entity_text = ""

                if not story_entities.empty:
                    people_text = join_unique(
                        story_entities[
                            story_entities["entity_type"].eq("person")
                        ]["entity"],
                        limit=4
                    )
                    entity_text = join_unique(
                        story_entities[
                            ~story_entities["entity_type"].eq("person")
                        ]["entity"],
                        limit=4
                    )

                first_published = first_row.get("first_published_at")
                if pd.isna(first_published):
                    first_published = first_row.get("published_at")

                story_rows.append({
                    "cluster_id": cluster_id,
                    "First published": first_published,
                    "First headline": first_row.get("title"),
                    "Articles": len(article_ids),
                    "Partners": cluster_df["media"].nunique(),
                    "Social posts": (
                        story_social["social_post_id"].nunique()
                        if (
                            not story_social.empty
                            and "social_post_id" in story_social.columns
                        )
                        else len(story_social)
                    ),
                    "Cluster confidence": first_row.get(
                        "cluster_confidence"
                    ),
                    "Topics": (
                        join_unique(story_topics["topic"], limit=4)
                        if not story_topics.empty
                        else ""
                    ),
                    "People": people_text,
                    "Other entities": entity_text,
                })

            story_index = (
                pd.DataFrame(story_rows)
                .sort_values("First published", ascending=False)
                .reset_index(drop=True)
            )

            st.caption(
                "Each row is one story. Click a row to open its full "
                "cross-partner and social distribution journey."
            )

            story_display = story_index[
                [
                    "First published",
                    "First headline",
                    "Articles",
                    "Partners",
                    "Social posts",
                    "Cluster confidence",
                    "Topics",
                    "People",
                    "Other entities",
                ]
            ].copy()

            story_event = st.dataframe(
                story_display,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="story_journeys_index",
                column_config={
                    "First published": st.column_config.DatetimeColumn(
                        "First published",
                        format="YYYY-MM-DD HH:mm"
                    ),
                    "First headline": "First headline",
                    "Articles": st.column_config.NumberColumn(
                        "Articles",
                        format="%d"
                    ),
                    "Partners": st.column_config.NumberColumn(
                        "Partners",
                        format="%d"
                    ),
                    "Social posts": st.column_config.NumberColumn(
                        "Social posts",
                        format="%d"
                    ),
                    "Cluster confidence": st.column_config.NumberColumn(
                        "Cluster confidence",
                        format="%.2f"
                    ),
                }
            )

            selected_story_rows = []
            try:
                selected_story_rows = list(story_event.selection.rows)
            except Exception:
                selected_story_rows = []

            if not selected_story_rows:
                st.info("Select a story row above to see the full journey.")

            else:
                selected_cluster_id = story_index.iloc[
                    selected_story_rows[0]
                ]["cluster_id"]

                selected_story = (
                    story_scope[
                        story_scope["cluster_id"].eq(selected_cluster_id)
                    ]
                    .copy()
                    .sort_values("published_at")
                )

                first_row = selected_story.iloc[0]
                story_article_ids = set(
                    selected_story["article_id"].dropna()
                )

                st.divider()
                st.markdown(
                    f"### {first_row.get('cluster_title', first_row.get('title', 'Story'))}"
                )

                if (
                    "cluster_summary" in selected_story.columns
                    and pd.notna(first_row.get("cluster_summary"))
                ):
                    st.write(first_row.get("cluster_summary"))

                story_perf = article_performance[
                    article_performance["article_id"].isin(
                        story_article_ids
                    )
                ].copy()

                story_social_latest = (
                    social_latest[
                        social_latest["article_id"].isin(story_article_ids)
                    ].copy()
                    if (
                        not social_latest.empty
                        and "article_id" in social_latest.columns
                    )
                    else pd.DataFrame()
                )

                story_social_posts_count = (
                    story_social_latest["target_id"].nunique()
                    if not story_social_latest.empty
                    else 0
                )

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric(
                    "Articles",
                    selected_story["article_id"].nunique()
                )
                c2.metric(
                    "Partners",
                    selected_story["media"].nunique()
                )
                c3.metric(
                    "Social posts",
                    story_social_posts_count
                )
                c4.metric(
                    "Website views",
                    compact_number(
                        sum_metric(story_perf, "website_views")
                    )
                )
                c5.metric(
                    "Social reach",
                    compact_number(
                        sum_metric(story_perf, "social_reach")
                    )
                )
                c6.metric(
                    "Cluster confidence",
                    (
                        f"{float(first_row['cluster_confidence']):.2f}"
                        if (
                            "cluster_confidence" in selected_story.columns
                            and pd.notna(first_row.get("cluster_confidence"))
                        )
                        else "—"
                    )
                )

                story_topics = (
                    topic_expanded[
                        topic_expanded["article_id"].isin(story_article_ids)
                    ].copy()
                    if not topic_expanded.empty
                    else pd.DataFrame()
                )

                story_entities = (
                    entity_expanded[
                        entity_expanded["article_id"].isin(story_article_ids)
                    ].copy()
                    if not entity_expanded.empty
                    else pd.DataFrame()
                )

                meta_left, meta_right = st.columns(2)

                with meta_left:
                    st.markdown("**Topics**")
                    st.write(
                        join_unique(story_topics["topic"], limit=10)
                        if not story_topics.empty
                        else "—"
                    )

                    st.markdown("**People**")
                    st.write(
                        join_unique(
                            story_entities[
                                story_entities["entity_type"].eq("person")
                            ]["entity"],
                            limit=10
                        )
                        if not story_entities.empty
                        else "—"
                    )

                with meta_right:
                    st.markdown("**Organizations, locations and events**")
                    st.write(
                        join_unique(
                            story_entities[
                                ~story_entities["entity_type"].eq("person")
                            ]["entity"],
                            limit=12
                        )
                        if not story_entities.empty
                        else "—"
                    )
                    st.caption(
                        "Audience figures below are summed across latest "
                        "platform snapshots and are not deduplicated users."
                    )

                # -----------------------------------------------------
                # Chronological journey table
                # -----------------------------------------------------
                st.markdown("#### Who published what, where and when")

                journey_rows = []

                for _, row in selected_story.iterrows():
                    metrics = article_metrics(row["article_id"])

                    journey_rows.append({
                        "Published": row.get("published_at"),
                        "Partner": row.get("media"),
                        "Channel": "Website",
                        "Type": "Article",
                        "Content": row.get("title"),
                        "Linked article": row.get("title"),
                        "Views": metrics.get("website_views", 0),
                        "Reach": np.nan,
                        "Audience": metrics.get("website_views", 0),
                        "Clicks": np.nan,
                        "URL": row.get("article_url"),
                    })

                if not story_social_latest.empty:
                    for _, row in story_social_latest.iterrows():
                        reach = pd.to_numeric(
                            pd.Series([row.get("reach_count")]),
                            errors="coerce"
                        ).fillna(0).iloc[0]
                        views = pd.to_numeric(
                            pd.Series([row.get("views_count")]),
                            errors="coerce"
                        ).fillna(0).iloc[0]
                        audience = reach if reach > 0 else views

                        journey_rows.append({
                            "Published": row.get("post_published_at"),
                            "Partner": row.get("media"),
                            "Channel": str(row.get("platform", "social")).title(),
                            "Type": "Social post",
                            "Content": row.get("post_text"),
                            "Linked article": row.get("article_title"),
                            "Views": views,
                            "Reach": reach,
                            "Audience": audience,
                            "Clicks": row.get("clicks_count", 0),
                            "URL": row.get("post_url"),
                        })

                journey_df = (
                    pd.DataFrame(journey_rows)
                    .sort_values("Published")
                    .reset_index(drop=True)
                )

                st.dataframe(
                    journey_df[
                        [
                            "Published",
                            "Partner",
                            "Channel",
                            "Content",
                            "Views",
                            "Reach",
                            "Clicks",
                            "URL",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Published": st.column_config.DatetimeColumn(
                            "Published",
                            format="YYYY-MM-DD HH:mm"
                        ),
                        "Partner": "Partner",
                        "Channel": "Channel",
                        "Content": "Publication / post",
                        "Views": st.column_config.NumberColumn(
                            "Views",
                            format="%d"
                        ),
                        "Reach": st.column_config.NumberColumn(
                            "Reach",
                            format="%d"
                        ),
                        "Clicks": st.column_config.NumberColumn(
                            "Clicks",
                            format="%d"
                        ),
                        "URL": st.column_config.LinkColumn("Open"),
                    }
                )

                # -----------------------------------------------------
                # Visual journey: bubble size = audience
                # -----------------------------------------------------
                st.markdown("#### Story distribution map")
                st.caption(
                    "Bubble size represents website views or social reach. "
                    "Larger objects indicate wider observed distribution."
                )

                visual_df = journey_df.dropna(
                    subset=["Published", "Partner"]
                ).copy()

                if not visual_df.empty:
                    visual_df["Bubble audience"] = (
                        pd.to_numeric(
                            visual_df["Audience"],
                            errors="coerce"
                        )
                        .fillna(0)
                        .clip(lower=1)
                    )

                    fig = px.scatter(
                        visual_df,
                        x="Published",
                        y="Partner",
                        size="Bubble audience",
                        color="Channel",
                        hover_name="Content",
                        hover_data={
                            "Views": ":,.0f",
                            "Reach": ":,.0f",
                            "Clicks": ":,.0f",
                            "Bubble audience": False,
                        },
                        size_max=55,
                        labels={
                            "Published": "Time",
                            "Partner": "Partner",
                            "Channel": "Channel",
                        }
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # -----------------------------------------------------
                # Compare distribution by partner and channel
                # -----------------------------------------------------
                st.markdown("#### Who distributed this story furthest?")

                distribution_rows = []

                for partner in sorted(selected_story["media"].dropna().unique()):
                    partner_perf = story_perf[
                        story_perf["media"].eq(partner)
                    ]
                    distribution_rows.append({
                        "Partner": partner,
                        "Channel": "Website views",
                        "Audience": sum_metric(
                            partner_perf,
                            "website_views"
                        )
                    })

                    if not story_social_latest.empty:
                        partner_social = story_social_latest[
                            story_social_latest["media"].eq(partner)
                        ]

                        for platform, pf in partner_social.groupby("platform"):
                            distribution_rows.append({
                                "Partner": partner,
                                "Channel": str(platform).title() + " reach",
                                "Audience": sum_metric(pf, "reach_count")
                            })

                distribution_df = pd.DataFrame(distribution_rows)

                if not distribution_df.empty:
                    fig = px.bar(
                        distribution_df,
                        x="Partner",
                        y="Audience",
                        color="Channel",
                        barmode="group",
                        labels={
                            "Audience": "Observed audience metric"
                        }
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # -----------------------------------------------------
                # Topic and entity spread
                # -----------------------------------------------------
                st.markdown("#### Which topics and mentions travelled furthest?")

                left, right = st.columns(2)

                with left:
                    st.markdown("##### Topics in this story")

                    topic_rows = []
                    if not story_topics.empty:
                        for topic, grp in story_topics.groupby("topic"):
                            ids = set(grp["article_id"].dropna())
                            perf = associated_performance(ids)
                            topic_rows.append({
                                "Topic": topic,
                                "Articles": len(ids),
                                "Partners": grp["media"].nunique(),
                                "Website views": perf["website_views"],
                                "Associated social reach": perf["social_reach"],
                            })

                    topic_spread = pd.DataFrame(topic_rows)

                    if not topic_spread.empty:
                        topic_spread = topic_spread.sort_values(
                            "Associated social reach",
                            ascending=False
                        )
                        st.dataframe(
                            topic_spread,
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No topic data for this story.")

                with right:
                    st.markdown("##### People and entities")

                    entity_rows = []
                    if not story_entities.empty:
                        for (entity, entity_type), grp in story_entities.groupby(
                            ["entity", "entity_type"]
                        ):
                            ids = set(grp["article_id"].dropna())
                            perf = associated_performance(ids)
                            entity_rows.append({
                                "Entity": entity,
                                "Type": entity_type,
                                "Mentions": (
                                    int(
                                        pd.to_numeric(
                                            grp.get("mentions_count", 1),
                                            errors="coerce"
                                        ).fillna(0).sum()
                                    )
                                    if "mentions_count" in grp.columns
                                    else len(grp)
                                ),
                                "Articles": len(ids),
                                "Partners": grp["media"].nunique(),
                                "Associated social reach": perf["social_reach"],
                            })

                    entity_spread = pd.DataFrame(entity_rows)

                    if not entity_spread.empty:
                        entity_spread = entity_spread.sort_values(
                            "Associated social reach",
                            ascending=False
                        )
                        st.dataframe(
                            entity_spread.head(15),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No entity data for this story.")

                if not story_entities.empty:
                    entity_chart_rows = []
                    for (entity, entity_type), grp in story_entities.groupby(
                        ["entity", "entity_type"]
                    ):
                        ids = set(grp["article_id"].dropna())
                        perf = associated_performance(ids)
                        entity_chart_rows.append({
                            "Entity": entity,
                            "Type": entity_type,
                            "Social reach": perf["social_reach"],
                        })

                    entity_chart = (
                        pd.DataFrame(entity_chart_rows)
                        .sort_values("Social reach", ascending=False)
                        .head(12)
                    )

                    if not entity_chart.empty:
                        fig = px.bar(
                            entity_chart.sort_values("Social reach"),
                            x="Social reach",
                            y="Entity",
                            color="Type",
                            orientation="h",
                            labels={
                                "Social reach": "Associated social reach"
                            }
                        )
                        fig.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 7. SOCIAL DISTRIBUTION
# =========================================================

with tabs[6]:

    st.subheader("Social distribution")

    if social_posts.empty:

        st.info("No social posts available.")

    else:

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Social posts",
            len(social_posts)
        )

        s2.metric(
            "Articles linked to social",
            len(
                set(
                    social_links[
                        "article_id"
                    ]
                )
                if not social_links.empty
                else set()
            )
        )

        coverage = (
            100
            * len(
                set(
                    social_links[
                        "article_id"
                    ]
                )
                if not social_links.empty
                else set()
            )
            / max(
                len(articles),
                1
            )
        )

        s3.metric(
            "Article linking coverage",
            f"{coverage:.1f}%"
        )

        if "platform" in social_posts.columns:

            by_platform = (
                social_posts
                .groupby(
                    "platform",
                    dropna=False
                )
                .size()
                .reset_index(
                    name="posts"
                )
                .sort_values(
                    "posts",
                    ascending=False
                )
            )

            st.markdown(
                "#### Posts by platform"
            )

            fig = px.bar(
                by_platform,
                x="platform",
                y="posts",
                labels={
                    "platform": "Platform",
                    "posts": "Posts"
                }
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if (
            "media" in social_posts.columns
            and "platform" in social_posts.columns
        ):

            by_partner_platform = (
                social_posts
                .groupby(
                    [
                        "media",
                        "platform"
                    ]
                )
                .size()
                .reset_index(
                    name="posts"
                )
            )

            st.markdown(
                "#### Social output by partner"
            )

            fig = px.bar(
                by_partner_platform,
                x="media",
                y="posts",
                color="platform",
                barmode="group",
                labels={
                    "media": "Partner",
                    "posts": "Posts",
                    "platform": "Platform"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            "#### Website ↔ Telegram links"
        )

        if article_social_view.empty:
            st.info(
                "No article ↔ social links are available."
            )

        else:
            telegram_view = article_social_view[
                article_social_view[
                    "platform"
                ].eq("telegram")
            ].copy()

            if selected_media:
                telegram_view = telegram_view[
                    telegram_view[
                        "media"
                    ].isin(
                        selected_media
                    )
                ]

            if telegram_view.empty:
                st.info(
                    "No Telegram links match the current filters."
                )

            else:
                t1, t2, t3 = st.columns(3)

                t1.metric(
                    "Website ↔ Telegram links",
                    len(
                        telegram_view
                    )
                )

                t2.metric(
                    "Linked website articles",
                    telegram_view[
                        "article_id"
                    ].nunique()
                )

                t3.metric(
                    "Partners represented",
                    telegram_view[
                        "media"
                    ].nunique()
                )

                by_partner = (
                    telegram_view
                    .groupby("media")
                    .size()
                    .reset_index(
                        name="linked_posts"
                    )
                    .sort_values(
                        "linked_posts",
                        ascending=False
                    )
                )

                fig = px.bar(
                    by_partner,
                    x="media",
                    y="linked_posts",
                    labels={
                        "media": "Partner",
                        "linked_posts": "Website ↔ Telegram links"
                    }
                )

                fig.update_layout(
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                show_cols = [
                    c for c in [
                        "article_published_at",
                        "media",
                        "title",
                        "article_url",
                        "published_at",
                        "post_text",
                        "post_url",
                        "relation_method",
                        "relation_confidence",
                    ]
                    if c in telegram_view.columns
                ]

                st.dataframe(
                    telegram_view[
                        show_cols
                    ].sort_values(
                        "article_published_at",
                        ascending=False
                    ),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "article_published_at":
                            st.column_config.DatetimeColumn(
                                "Website published",
                                format="YYYY-MM-DD HH:mm"
                            ),
                        "media": "Partner",
                        "title": "Website article",
                        "article_url":
                            st.column_config.LinkColumn(
                                "Open website"
                            ),
                        "published_at":
                            st.column_config.DatetimeColumn(
                                "Telegram published",
                                format="YYYY-MM-DD HH:mm"
                            ),
                        "post_text": "Telegram post",
                        "post_url":
                            st.column_config.LinkColumn(
                                "Open Telegram"
                            ),
                        "relation_method": "Link method",
                        "relation_confidence":
                            st.column_config.NumberColumn(
                                "Confidence",
                                format="%.2f"
                            ),
                    },
                )

        st.markdown(
            "#### Recent social posts"
        )

        social_cols = [
            c for c in [
                "published_at",
                "media",
                "platform",
                "post_text",
                "post_url"
            ]
            if c in social_posts.columns
        ]

        social_show = (
            social_posts[
                social_cols
            ]
            .sort_values(
                "published_at",
                ascending=False
            )
            if "published_at"
            in social_posts.columns
            else social_posts[
                social_cols
            ]
        )

        st.dataframe(
            social_show.head(300),
            use_container_width=True,
            hide_index=True,
            column_config={
                "post_url":
                    st.column_config.LinkColumn(
                        "Open post"
                    )
            }
        )



# =========================================================
# 8. PERFORMANCE
# =========================================================

with tabs[7]:

    st.subheader("Performance")

    st.caption(
        "Latest cumulative metric snapshot for each article and social post. "
        "Historical snapshots are kept separately so growth over time can "
        "also be shown."
    )

    if metrics_snapshots.empty:
        st.info("No performance metrics are available.")

    else:
        perf_web = (
            web_latest[
                web_latest["article_id"].isin(filtered_article_ids)
            ].copy()
            if not web_latest.empty
            else pd.DataFrame()
        )

        perf_social = (
            social_latest[
                social_latest["article_id"].isin(filtered_article_ids)
            ].copy()
            if (
                not social_latest.empty
                and "article_id" in social_latest.columns
            )
            else pd.DataFrame()
        )

        website_views = sum_metric(
            perf_web,
            "views_count"
        )
        social_views = sum_metric(
            perf_social,
            "views_count"
        )
        social_reach = sum_metric(
            perf_social,
            "reach_count"
        )
        interactions = social_interactions(
            perf_social
        )
        clicks = sum_metric(
            perf_social,
            "clicks_count"
        )

        avg_engagement = 0
        if (
            not perf_web.empty
            and "avg_engagement_seconds" in perf_web.columns
        ):
            avg_engagement = (
                pd.to_numeric(
                    perf_web["avg_engagement_seconds"],
                    errors="coerce"
                )
                .dropna()
                .mean()
            )

        a, b, c, d, e, f = st.columns(6)

        a.metric(
            "Website views",
            compact_number(website_views)
        )
        b.metric(
            "Social views",
            compact_number(social_views)
        )
        c.metric(
            "Social reach",
            compact_number(social_reach)
        )
        d.metric(
            "Interactions",
            compact_number(interactions)
        )
        e.metric(
            "Link clicks",
            compact_number(clicks)
        )
        f.metric(
            "Avg web engagement",
            (
                f"{avg_engagement:.0f}s"
                if pd.notna(avg_engagement)
                else "—"
            )
        )

        st.divider()

        # -------------------------------------------------
        # Coverage
        # -------------------------------------------------

        left, right = st.columns(2)

        with left:
            st.markdown(
                "#### Website → social coverage"
            )

            coverage_rows = []

            for platform in [
                "telegram",
                "facebook",
                "instagram"
            ]:
                if article_social_view.empty:
                    linked = 0
                else:
                    tmp = article_social_view[
                        article_social_view[
                            "article_id"
                        ].isin(filtered_article_ids)
                    ]

                    tmp = tmp[
                        tmp["platform"].eq(platform)
                    ]

                    linked = tmp[
                        "article_id"
                    ].nunique()

                coverage_rows.append({
                    "platform": platform.title(),
                    "coverage_pct": (
                        100 * linked
                        / max(
                            len(filtered_article_ids),
                            1
                        )
                    ),
                    "linked_articles": linked,
                })

            coverage_df = pd.DataFrame(
                coverage_rows
            )

            fig = px.bar(
                coverage_df,
                x="platform",
                y="coverage_pct",
                text="coverage_pct",
                hover_data=[
                    "linked_articles"
                ],
                labels={
                    "platform": "Platform",
                    "coverage_pct": "Articles distributed (%)"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.0f}%",
                textposition="outside"
            )
            fig.update_yaxes(
                range=[0, 100]
            )
            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            st.markdown(
                "#### Final social reach by platform"
            )

            if not perf_social.empty:
                platform_perf = (
                    perf_social
                    .groupby("platform")
                    .agg(
                        reach=("reach_count", "sum"),
                        views=("views_count", "sum"),
                        clicks=("clicks_count", "sum")
                    )
                    .reset_index()
                )

                fig = px.bar(
                    platform_perf,
                    x="platform",
                    y="reach",
                    labels={
                        "platform": "Platform",
                        "reach": "Reach"
                    }
                )
                fig.update_layout(
                    showlegend=False
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
            else:
                st.info(
                    "No social metrics match the current filters."
                )

        # -------------------------------------------------
        # Partner performance
        # -------------------------------------------------

        st.markdown(
            "#### Performance by partner"
        )

        partner_rows = []

        partner_names = sorted(
            filtered_articles[
                "media"
            ].dropna().unique()
        )

        for partner in partner_names:
            partner_article_ids = set(
                filtered_articles[
                    filtered_articles[
                        "media"
                    ].eq(partner)
                ]["id"]
            )

            w = (
                perf_web[
                    perf_web["article_id"].isin(
                        partner_article_ids
                    )
                ]
                if not perf_web.empty
                else pd.DataFrame()
            )

            s = (
                perf_social[
                    perf_social["article_id"].isin(
                        partner_article_ids
                    )
                ]
                if not perf_social.empty
                else pd.DataFrame()
            )

            partner_rows.append({
                "Partner": partner,
                "Website views": sum_metric(
                    w,
                    "views_count"
                ),
                "Social reach": sum_metric(
                    s,
                    "reach_count"
                ),
                "Social views": sum_metric(
                    s,
                    "views_count"
                ),
            })

        partner_perf = pd.DataFrame(
            partner_rows
        )

        if not partner_perf.empty:
            partner_long = partner_perf.melt(
                id_vars=["Partner"],
                var_name="Metric",
                value_name="Value"
            )

            fig = px.bar(
                partner_long,
                x="Partner",
                y="Value",
                color="Metric",
                barmode="group"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # -------------------------------------------------
        # Top website articles
        # -------------------------------------------------

        st.markdown(
            "#### Top-performing website articles"
        )

        if not perf_web.empty:
            top_web = perf_web.copy()

            social_agg = pd.DataFrame()

            if not perf_social.empty:
                social_agg = (
                    perf_social
                    .groupby("article_id")
                    .agg(
                        social_reach=("reach_count", "sum"),
                        social_views=("views_count", "sum"),
                        social_clicks=("clicks_count", "sum"),
                        social_posts=("target_id", "nunique")
                    )
                    .reset_index()
                )

                top_web = top_web.merge(
                    social_agg,
                    on="article_id",
                    how="left"
                )

            for col in [
                "social_reach",
                "social_views",
                "social_clicks",
                "social_posts"
            ]:
                if col not in top_web.columns:
                    top_web[col] = 0

            top_web = top_web.sort_values(
                "views_count",
                ascending=False
            ).head(20)

            st.dataframe(
                top_web[
                    [
                        c for c in [
                            "media",
                            "title",
                            "views_count",
                            "users_count",
                            "avg_engagement_seconds",
                            "social_posts",
                            "social_reach",
                            "social_views",
                            "social_clicks",
                            "article_url",
                        ]
                        if c in top_web.columns
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "media": "Partner",
                    "title": "Article",
                    "views_count":
                        st.column_config.NumberColumn(
                            "Web views",
                            format="%d"
                        ),
                    "users_count":
                        st.column_config.NumberColumn(
                            "Users",
                            format="%d"
                        ),
                    "avg_engagement_seconds":
                        st.column_config.NumberColumn(
                            "Avg engagement (s)",
                            format="%.0f"
                        ),
                    "social_posts":
                        st.column_config.NumberColumn(
                            "Linked social posts",
                            format="%d"
                        ),
                    "social_reach":
                        st.column_config.NumberColumn(
                            "Social reach",
                            format="%d"
                        ),
                    "social_views":
                        st.column_config.NumberColumn(
                            "Social views",
                            format="%d"
                        ),
                    "social_clicks":
                        st.column_config.NumberColumn(
                            "Clicks",
                            format="%d"
                        ),
                    "article_url":
                        st.column_config.LinkColumn(
                            "Open"
                        ),
                }
            )

        st.divider()

        # -------------------------------------------------
        # Article → social journey
        # -------------------------------------------------

        st.markdown(
            "#### Article → social performance journey"
        )

        article_choices = (
            filtered_articles[
                [
                    "id",
                    "media",
                    "title",
                    "published_at"
                ]
            ]
            .sort_values(
                "published_at",
                ascending=False
            )
            .copy()
        )

        article_choices["label"] = (
            article_choices["media"].astype(str)
            + " — "
            + article_choices["title"].astype(str)
        )

        if len(article_choices):
            selected_label = st.selectbox(
                "Choose a website article",
                article_choices["label"].tolist(),
                key="performance_article_selector"
            )

            chosen = article_choices[
                article_choices["label"].eq(
                    selected_label
                )
            ].iloc[0]

            chosen_id = int(
                chosen["id"]
            )

            chosen_web = (
                perf_web[
                    perf_web["article_id"].eq(
                        chosen_id
                    )
                ]
                if not perf_web.empty
                else pd.DataFrame()
            )

            chosen_social = (
                perf_social[
                    perf_social["article_id"].eq(
                        chosen_id
                    )
                ].copy()
                if not perf_social.empty
                else pd.DataFrame()
            )

            j1, j2, j3, j4 = st.columns(4)

            j1.metric(
                "Website views",
                compact_number(
                    sum_metric(
                        chosen_web,
                        "views_count"
                    )
                )
            )
            j2.metric(
                "Social posts",
                (
                    chosen_social["target_id"].nunique()
                    if not chosen_social.empty
                    else 0
                )
            )
            j3.metric(
                "Social reach",
                compact_number(
                    sum_metric(
                        chosen_social,
                        "reach_count"
                    )
                )
            )
            j4.metric(
                "Clicks",
                compact_number(
                    sum_metric(
                        chosen_social,
                        "clicks_count"
                    )
                )
            )

            if not chosen_social.empty:
                journey = chosen_social.copy()

                if (
                    "post_published_at" in journey.columns
                    and pd.notna(
                        chosen["published_at"]
                    )
                ):
                    journey["delay_minutes"] = (
                        journey["post_published_at"]
                        - chosen["published_at"]
                    ).dt.total_seconds() / 60

                st.dataframe(
                    journey[
                        [
                            c for c in [
                                "platform",
                                "post_published_at",
                                "delay_minutes",
                                "views_count",
                                "reach_count",
                                "likes_count",
                                "reactions_count",
                                "comments_count",
                                "shares_count",
                                "forwards_count",
                                "clicks_count",
                                "post_url",
                            ]
                            if c in journey.columns
                        ]
                    ].sort_values(
                        "post_published_at"
                    ),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "platform": "Platform",
                        "post_published_at":
                            st.column_config.DatetimeColumn(
                                "Published",
                                format="YYYY-MM-DD HH:mm"
                            ),
                        "delay_minutes":
                            st.column_config.NumberColumn(
                                "Delay from website (min)",
                                format="%.0f"
                            ),
                        "post_url":
                            st.column_config.LinkColumn(
                                "Open post"
                            ),
                    }
                )

            # Historical growth chart for website + linked social posts.
            history_rows = []

            article_history = metrics_snapshots[
                metrics_snapshots[
                    "target_type"
                ].eq("article")
                & metrics_snapshots[
                    "target_id"
                ].eq(chosen_id)
            ].copy()

            if not article_history.empty:
                for _, r in article_history.iterrows():
                    history_rows.append({
                        "snapshot_at": r[
                            "snapshot_at"
                        ],
                        "series": "Website views",
                        "value": r.get(
                            "views_count",
                            0
                        )
                    })

            linked_post_ids = (
                chosen_social[
                    "target_id"
                ].dropna().tolist()
                if not chosen_social.empty
                else []
            )

            for post_id in linked_post_ids:
                post_history = metrics_snapshots[
                    metrics_snapshots[
                        "target_type"
                    ].eq("social_post")
                    & metrics_snapshots[
                        "target_id"
                    ].eq(post_id)
                ].copy()

                if post_history.empty:
                    continue

                platform = (
                    post_history[
                        "platform"
                    ].iloc[0]
                )

                preferred_metric = (
                    "reach_count"
                    if (
                        platform
                        in [
                            "facebook",
                            "instagram"
                        ]
                        and "reach_count"
                        in post_history.columns
                    )
                    else "views_count"
                )

                for _, r in post_history.iterrows():
                    history_rows.append({
                        "snapshot_at": r[
                            "snapshot_at"
                        ],
                        "series": (
                            f"{str(platform).title()} "
                            + (
                                "reach"
                                if preferred_metric
                                == "reach_count"
                                else "views"
                            )
                        ),
                        "value": r.get(
                            preferred_metric,
                            0
                        )
                    })

            history = pd.DataFrame(
                history_rows
            )

            if not history.empty:
                st.markdown(
                    "##### Growth after publication"
                )

                fig = px.line(
                    history.sort_values(
                        "snapshot_at"
                    ),
                    x="snapshot_at",
                    y="value",
                    color="series",
                    markers=True,
                    labels={
                        "snapshot_at": "Time",
                        "value": "Cumulative metric",
                        "series": "Channel"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


st.caption(
    "Demo: Media Partners shared intelligence layer. "
    "All August 2026 content and performance figures in this environment "
    "are synthetic. Same-story clusters, direct citations and possible "
    "pickups remain separate evidence types."
)
