import streamlit as st
import altair as alt
import pandas as pd  

st.title("SERO – Usage patterns & crisis-related activity")

# Check that the data has been loaded via the "Load data" page.
# If not, we stop the script and show an error message.
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded yet. Please go to the **Load data** page first.")
    st.stop()

data = st.session_state.sero_data
events = data["events"]
careplan = data["careplan"]
supportcareplan = data["supportcareplan"]
observations = data["observations"]
qr = data["qr"]
all_interactions = data["all_interactions"]


# ---------------------------------------------------------------------
# SIDEBAR: GLOBAL FILTER
# ---------------------------------------------------------------------
# The sidebar controls which subset of the SERO dataset is used
# for all visualisations on this page:
# - date range
# - sources (events, plans, observations, questionnaires)
# - day period (Morning / Afternoon / Evening/Night)
# - optional filter on a single anonymised user.
with st.sidebar:
    st.header("Filters")

    # Global date range filter based on all interactions
    min_date = all_interactions["date"].min()
    max_date = all_interactions["date"].max()
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

    # Interaction source type (event, careplan, support plan, observation, questionnaire)
    # We include all sources by default and do not expose a specific filter in the UI.
    available_sources = sorted(all_interactions["source"].unique())
    selected_sources = available_sources
    st.caption("All interaction sources are included in this page (no source filter applied).")

    # Filter on part of the day (Morning / Afternoon / Evening/Night)
    available_periods = ["Morning", "Afternoon", "Evening/Night"]
    selected_periods = st.multiselect(
        "Day periods",
        options=available_periods,
        default=available_periods,
    )

    # Optional: filter on a single anonymised user (iduser)
    all_subjects = sorted(all_interactions["iduser"].dropna().unique())
    subject_filter = st.selectbox(
        "Filter by anonymised user (optional)",
        options=["(all)"] + all_subjects,
        index=0,
    )

# Apply the global filters (date, source, day period, user) to build
# a boolean mask and derive the filtered interaction dataframe.
mask = (
    (all_interactions["date"] >= start_date)
    & (all_interactions["date"] <= end_date)
    & (all_interactions["source"].isin(selected_sources))
    & (all_interactions["day_period"].isin(selected_periods))
)

if subject_filter != "(all)":
    mask &= all_interactions["iduser"].eq(subject_filter)

filtered_int = all_interactions[mask]

st.markdown(
    f"**Filtered interactions:** {len(filtered_int)} "
    f"(from {start_date} to {end_date})"
)


# ---------------------------------------------------------------------
#1. GLOBAL – WHEN DO USERS MOSTLY USE THE APP?
# ---------------------------------------------------------------------
# This section answers:
# - At which hours of the day SERO is used,
# - On which weekdays interactions happen more often,
# - How usage is distributed across Morning / Afternoon / Evening/Night.
st.header("1. Global usage – When are users mostly using the app?")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Unique users (filtered)", filtered_int["iduser"].nunique())
with col2:
    st.metric("Total interactions", len(filtered_int))
with col3:
    st.metric("Sources included", ", ".join(selected_sources) or "none")

# a) Distribution over 24 hours (hour of the day)
st.subheader("Usage over 24 hours")

hour_counts = (
    filtered_int.groupby("hour", observed=True)
    .size()
    .rename("count")
    .reset_index()
    .set_index("hour")
)

st.bar_chart(hour_counts)

# b) Weekly pattern – distribution of interactions by weekday
st.subheader("Weekly pattern – all interactions")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

weekday_counts = (
    filtered_int["weekday"]
    .value_counts()
    .reindex(weekday_order)
    .fillna(0)
    .rename("count")
    .to_frame()
)

weekday_df = weekday_counts.reset_index().rename(columns={"index": "weekday"})

chart_weekday = (
    alt.Chart(weekday_df)
    .mark_bar()
    .encode(
        x=alt.X("weekday:N", sort=weekday_order, title="Weekday"),
        y=alt.Y("count:Q", title="Number of interactions")
    )
)

st.altair_chart(chart_weekday, width="stretch")

# c) Distribution of interactions by part of the day
#    (Morning / Afternoon / Evening/Night)
st.subheader("Distribution by day period (Morning / Afternoon / Evening/Night)")

period_order = ["Morning", "Afternoon", "Evening/Night"]

period_counts = (
    filtered_int["day_period"]
    .value_counts()
    .reindex(period_order)
    .fillna(0)
    .rename("count")
    .to_frame()
)

period_df = period_counts.reset_index().rename(columns={"index": "day_period"})

chart_period = (
    alt.Chart(period_df)
    .mark_bar()
    .encode(
        x=alt.X("day_period:N", sort=period_order, title="Day period"),
        y=alt.Y("count:Q", title="Number of interactions")
    )
)

st.altair_chart(chart_period, width="stretch")


# ---------------------------------------------------------------------
#2. CRISIS / SUICIDAL IDEATION – WEEKLY PATTERN 
# ---------------------------------------------------------------------
# This section focuses on questionnaire answers that contain suicidal ideation.
# The heuristic flag `is_suicidal` is computed at dataset build time
# (based on the presence of grammatical roots like 'suicide' / 'suizid').
# Here, we:
# - show the total number of questionnaire answers,
# - highlight how many are flagged as suicidal,
# - plot their distribution by weekday and by day period.
st.header("2. Weekly pattern of suicidal ideation entries (Questionnaire responses)")

# Filter questionnaire responses by the global date range and optional user
# so that crisis-related visualisations stay consistent with the filters.
qr_filtered = qr[
    (qr["date"] >= start_date)
    & (qr["date"] <= end_date)
]

if subject_filter != "(all)":
    qr_filtered = qr_filtered[qr_filtered["iduser"].eq(subject_filter)]

# Keep only the answers that are flagged as containing suicidal ideation.
qr_suicidal = qr_filtered[qr_filtered["is_suicidal"]]

st.write(
    f"- Total `questionnaire` answers in range: **{len(qr_filtered)}**  \n"
    f"- Responses flagged as containing suicidal ideation: **{len(qr_suicidal)}** "
    f"(if they contain the grammatical root of 'suicide' in English or German)"
)

if len(qr_suicidal) > 0:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    suicidal_weekday = (
        qr_suicidal["weekday"]
        .value_counts()
        .reindex(weekday_order)
        .fillna(0)
        .rename("count")
        .to_frame()
    )

    suicidal_weekday_df = (
        suicidal_weekday
        .reindex(weekday_order)
        .reset_index()
        .rename(columns={"index": "weekday"})
    )

    st.subheader("Weekly pattern – suicidal ideation answers")

    chart_suicidal_weekday = (
        alt.Chart(suicidal_weekday_df)
        .mark_bar()
        .encode(
            x=alt.X("weekday:N", sort=weekday_order, title="Weekday"),
            y=alt.Y("count:Q", title="Number of suicidal answers")
        )
    )

    st.altair_chart(chart_suicidal_weekday, width="stretch")

    period_order = ["Morning", "Afternoon", "Evening/Night"]

    suicidal_period = (
        qr_suicidal["day_period"]
        .value_counts()
        .reindex(period_order)
        .fillna(0)
        .rename("count")
        .to_frame()
    )

    suicidal_period_df = (
        suicidal_period
        .reindex(period_order)
        .reset_index()
        .rename(columns={"index": "day_period"})
    )

    st.subheader("Day-period pattern – suicidal ideation answers")

    chart_suicidal_period = (
        alt.Chart(suicidal_period_df)
        .mark_bar()
        .encode(
            x=alt.X("day_period:N", sort=period_order, title="Day period"),
            y=alt.Y("count:Q", title="Number of suicidal answers")
        )
    )

    st.altair_chart(chart_suicidal_period, width="stretch")

else:
    st.info("No questionnaire answer flagged as suicidal in the current filter.")


# ---------------------------------------------------------------------
# 3. EVENTS DURING THE DAY – TYPE & DAY PERIOD 
# ---------------------------------------------------------------------
# This section looks specifically at app events and answers:
# - How many events occur in the selected date range?
# - Which types of events dominate in each part of the day?
# We create a simplified event_type for readability (assessment, boards, emergency call, etc.).

st.header("3. Amount and type of events during the day")

events_mask = (
    (events["date"] >= start_date)
    & (events["date"] <= end_date)
    & (events["day_period"].isin(selected_periods))
)
if subject_filter != "(all)":
    events_mask &= events["iduser"].eq(subject_filter)

events_filtered = events[events_mask].copy()

st.write(f"Number of events in range: **{len(events_filtered)}**")

# Helper function to map raw event_category / event_name combinations
# into a smaller set of readable event types for the visualisations.
def label_event(row):
    if row["event_name"] == "optionCallEmergencyContact":
        return "CallEmergencyContact"
    elif row["event_category"] == "assessment":
        return "Assessment"
    elif row["event_category"] == "treasureChest":
        return "Board / TreasureChest"
    else:
        return f"{row['event_category']}/{row['event_name']}"

events_filtered["event_type"] = events_filtered.apply(label_event, axis=1)

# Pivot the counts so that day periods are rows and event types are columns.
# This makes it easy to inspect the distribution both in table and bar-chart form.
event_counts = (
    events_filtered
    .groupby(["day_period", "event_type"], observed=True)
    .size()
    .reset_index(name="n")
)

pivot_events = (
    event_counts
    .pivot(index="day_period", columns="event_type", values="n")
    .fillna(0)
    .reindex(index=["Morning", "Afternoon", "Evening/Night"])
)

st.subheader("Events per day period and type")
st.dataframe(pivot_events)

period_order = ["Morning", "Afternoon", "Evening/Night"]

pivot_long = (
    pivot_events
    .reindex(index=period_order)
    .reset_index()
    .melt(
        id_vars="day_period",
        var_name="event_type",
        value_name="count"
    )
)

chart_events = (
    alt.Chart(pivot_long)
    .mark_bar()
    .encode(
        x=alt.X("day_period:N", sort=period_order, title="Day period"),
        y=alt.Y("count:Q", title="Number of events"),
        color=alt.Color("event_type:N", title="Event type"),
        tooltip=["day_period", "event_type", "count"]
    )
)

st.altair_chart(chart_events, width="stretch")


# ---------------------------------------------------------------------
#  4. SMOOTH GLOBAL TREND SINCE BEGINNING 
# ---------------------------------------------------------------------
# This section computes a smoothed trend of global usage over time.
# We:
# - count the number of interactions per day,
# - compute a 15-day rolling mean,
# - sample one point every 15 days to make the line chart readable.
st.header("4. Smooth global usage trend since the beginning")

mask_full = (
    all_interactions["source"].isin(selected_sources)
    & all_interactions["day_period"].isin(selected_periods)
)

if subject_filter != "(all)":
    mask_full &= all_interactions["iduser"].eq(subject_filter)

full_int = all_interactions[mask_full].copy()

daily_counts = (
    full_int.groupby("date")
    .size()
    .rename("count")
    .reset_index()
)

daily_counts["date"] = pd.to_datetime(daily_counts["date"])
daily_counts = daily_counts.sort_values("date").set_index("date")

# 15-day rolling mean: for each day we average the previous 15 days
# of activity to smooth out short-term spikes.
daily_counts["mean_15d"] = (
    daily_counts["count"]
    .rolling(window=15, min_periods=1)
    .mean()
)

smooth_points = daily_counts.iloc[14::15].reset_index()
if smooth_points.empty:
    smooth_points = daily_counts.tail(1).reset_index()

st.write(
    f"Showing one point every 15 days, each being the mean usage of the previous 15 days. "
    f"Total points: **{len(smooth_points)}**"
)

chart_smooth = (
    alt.Chart(smooth_points)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("mean_15d:Q", title="Mean interactions over previous 15 days"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("mean_15d:Q", title="15-day mean", format=".2f"),
        ],
    )
    .properties(height=350)
)

st.altair_chart(chart_smooth, width="stretch")