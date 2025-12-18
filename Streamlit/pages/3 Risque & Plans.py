import streamlit as st
import altair as alt
import pandas as pd

# ---------------------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------------------
st.title("SERO – Risk & support plans")

# ---------------------------------------------------------------------
# CHECK THAT DATA IS LOADED
# ---------------------------------------------------------------------
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded yet. Please go to the **Load data** page first.")
    st.stop()

# Retrieve pre-built dataframes from the dataset created in the Load page
data = st.session_state.sero_data
events = data["events"]
careplan = data["careplan"]
supportcareplan = data["supportcareplan"]
observations = data["observations"]
qr = data["qr"]

# ---------------------------------------------------------------------
# SIDEBAR – GLOBAL FILTERS (DATE + USER)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Filters – Risk & plans")

    if observations.empty:
        st.warning("No observations available – risk-related analyses may be limited.")

    # Default date range is based on observations (they reflect crisis proximity).
    # Fallback to events if observations are empty.
    if not observations.empty:
        min_date = observations["date"].min()
        max_date = observations["date"].max()
    else:
        min_date = events["date"].min()
        max_date = events["date"].max()

    # Date range filter applied to all analyses on this page
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Build the list of users based on observations + plans
    # (these are the actors in risk & care workflows)
    user_ids = set()
    if "iduser" in observations.columns:
        user_ids |= set(observations["iduser"].dropna().unique())
    if "iduser" in careplan.columns:
        user_ids |= set(careplan["iduser"].dropna().unique())
    if "iduser" in supportcareplan.columns:
        user_ids |= set(supportcareplan["iduser"].dropna().unique())

    all_users = sorted(user_ids)
    user_filter = st.selectbox(
        "Filter by anonymised user (optional)",
        options=["(all)"] + all_users,
        index=0,
    )

# Intro text explaining what this page covers
st.markdown(
    "This page focuses on two main questions:\n"
    "4. **Distribution & trend of distance** – How does perceived crisis proximity evolve?  \n"
    "5. **Plan themes** – What type of support is proposed to users and supporters?"
)

# ---------------------------------------------------------------------
# HELPER – APPLY DATE + USER FILTER TO A DATAFRAME
# ---------------------------------------------------------------------
def apply_date_user_filter(
    df: pd.DataFrame, start_date, end_date, user_filter: str
) -> pd.DataFrame:
    """
    Apply the global date range and optional user filter to a dataframe
    that contains at least a `date` column and (optionally) an `iduser` column.
    Returns a filtered copy.
    """
    if df.empty:
        return df

    # Filter on date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    # Filter on user if a specific anonymised user is chosen
    if user_filter != "(all)" and "iduser" in df.columns:
        mask &= df["iduser"].eq(user_filter)

    return df[mask].copy()

# ---------------------------------------------------------------------
# 4. DISTRIBUTION & TREND OF DISTANCE (CRISIS PROXIMITY)
# ---------------------------------------------------------------------
st.header("4. Proximity of crisis – distribution & evolution of distance")

# Apply filters to observations (these contain the distance between circles)
obs_filtered = apply_date_user_filter(observations, start_date, end_date, user_filter)

if obs_filtered.empty or "distance" not in obs_filtered.columns:
    st.info("No observations with distance available for the current filters.")
else:
    # Ensure distance is numeric so that Altair can bin the values correctly
    obs_filtered["distance"] = pd.to_numeric(obs_filtered["distance"], errors="coerce")

    # ------------------------------------------------------------------
    # 4.1 Distribution of distances (how far from crisis?)
    # ------------------------------------------------------------------
    st.subheader("4.1 Distribution of distances (how far from crisis?)")

    hist_chart = (
        alt.Chart(obs_filtered.dropna(subset=["distance"]))
        .mark_bar()
        .encode(
            x=alt.X(
                "distance:Q",
                bin=alt.Bin(maxbins=30),
                title="Distance between circles",
            ),
            y=alt.Y("count():Q", title="Number of observations"),
            tooltip=[alt.Tooltip("count():Q", title="Observations")],
        )
    )
    st.altair_chart(hist_chart, width="stretch")

    # ------------------------------------------------------------------
    # 4.2 Weekly trend of perceived crisis distance (median)
    # ------------------------------------------------------------------
    st.subheader("4.2 Weekly trend of perceived crisis distance (median)")

    # Convert the observation date to a weekly period, then to the start of the week.
    obs_filtered["week"] = (
        pd.to_datetime(obs_filtered["date"])
        .dt.to_period("W")
        .dt.start_time
    )

    # Compute the median distance per week to summarise perceived crisis proximity.
    weekly_median = (
        obs_filtered.groupby("week")["distance"]
        .median()
        .reset_index()
        .sort_values("week")
    )

    if weekly_median.empty:
        st.info("Not enough data to compute a weekly trend.")
    else:
        chart_trend = (
            alt.Chart(weekly_median)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:T", title="Week"),
                y=alt.Y("distance:Q", title="Median distance"),
                tooltip=[
                    alt.Tooltip("week:T", title="Week"),
                    alt.Tooltip("distance:Q", title="Median distance", format=".2f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_trend, width="stretch")

        st.caption(
            "Lower distance values correspond to feeling closer to crisis. "
            "The curve shows how the median perceived proximity to crisis evolves over time "
            "for the selected date range and (optionally) user."
        )

# ---------------------------------------------------------------------
# 5. PLAN THEMES – USER VS SUPPORTER
# ---------------------------------------------------------------------
st.header("5. Plan themes – user vs supporter")

# Apply the date + user filters to both types of care plans
careplan_filt = apply_date_user_filter(careplan, start_date, end_date, user_filter)
support_filt = apply_date_user_filter(supportcareplan, start_date, end_date, user_filter)

if careplan_filt.empty and support_filt.empty:
    st.info("No care plan entries for the selected filters.")
else:
    # We will build a unified dataframe with:
    #   - topic    : textual theme (e.g., 'Talk to someone', 'Creativity', etc.)
    #   - count    : how many times this topic appears
    #   - plan_type: 'User' vs 'Supporter'
    dfs_topics = []

    # ----- User plans (careplan) -----
    if not careplan_filt.empty and "topic" in careplan_filt.columns:
        cp_topics = (
            careplan_filt["topic"]
            .dropna()
            .astype(str)
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "topic"})
        )
        cp_topics["plan_type"] = "User"
        cp_topics = cp_topics[["topic", "count", "plan_type"]]
        dfs_topics.append(cp_topics)

    # ----- Supporter plans (supportcareplan) -----
    if not support_filt.empty and "topic" in support_filt.columns:
        sp_topics = (
            support_filt["topic"]
            .dropna()
            .astype(str)
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "topic"})
        )
        sp_topics["plan_type"] = "Supporter"
        sp_topics = sp_topics[["topic", "count", "plan_type"]]
        dfs_topics.append(sp_topics)

    if not dfs_topics:
        st.info("No topic information available in care plans.")
    else:
        # Concatenate user + supporter topics into one dataframe
        topics_all = pd.concat(dfs_topics, ignore_index=True)

        # Safety check: make sure the expected columns exist
        expected_cols = {"topic", "count", "plan_type"}
        missing_cols = expected_cols - set(topics_all.columns)
        if missing_cols:
            st.error(
                f"Unexpected structure in care plan topics (missing columns: {missing_cols}). "
                "Please check the input CSV column names (expected: 'topic')."
            )
        else:
            # We keep only the top 8 themes (based on total count user+supporter)
            # to keep the chart readable.
            top_topics_order = (
                topics_all.groupby("topic")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(8)
                .index
            )
            topics_all = topics_all[topics_all["topic"].isin(top_topics_order)]

            st.subheader("5.1 Most frequent plan themes for users vs supporters")

            # Grouped bar chart: x = theme, color = plan type (User/Supporter)
            topics_chart = (
                alt.Chart(topics_all)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "topic:N",
                        sort=list(topics_all["topic"].unique()),
                        title="Theme",
                    ),
                    y=alt.Y("count:Q", title="Number of occurrences"),
                    color=alt.Color("plan_type:N", title="Plan"),
                    tooltip=["topic", "plan_type", "count"],
                )
            )
            st.altair_chart(topics_chart, width="stretch")

            st.caption(
                "User plans typically describe what the person in distress can do (self-strategies), "
                "while supporter plans describe how others can help. Comparing the distributions "
                "helps to see whether both sides are aligned (e.g., both emphasise 'Talk to someone') "
                "or whether there is a mismatch in the type of support that is documented."
            )